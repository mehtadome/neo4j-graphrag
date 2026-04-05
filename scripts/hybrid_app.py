"""
GraphRAG app using custom parallel fusion of VectorCypherRetriever and Text2CypherRetriever.
Best for hybrid questions that require both relational context and aggregate data —
e.g. "Which agency drove the most revenue from customers who booked a demo?"

Routing logic:
  - Pure contextual questions  → VectorCypherRetriever only
  - Pure aggregation questions → Text2CypherRetriever only
  - Hybrid questions           → both retrievers in parallel, results merged before LLM

For isolated retriever testing use:
  - vector_app.py       (VectorCypherRetriever only)
  - text2cypher_app.py  (Text2CypherRetriever only)
  - graphrag_app.py     (auto-routing, no fusion)

Usage:
    python scripts/hybrid_app.py
"""

import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import VectorCypherRetriever, Text2CypherRetriever
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.types import RetrieverResultItem

load_dotenv()

NEO4J_URI      = os.environ["NEO4J_URI"]
NEO4J_USER     = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

# --- Routing keyword sets ---

AGGREGATION_KEYWORDS = {
    "most", "best", "highest", "lowest", "total", "count",
    "average", "top", "least", "rank", "how many", "sum",
    "which campaign", "which channel", "which product", "which agency",
}

# Contextual signals that, when combined with aggregation keywords, indicate a hybrid question
CONTEXTUAL_KEYWORDS = {
    "journey", "who", "customers who", "after", "before", "booked", "started",
    "that", "which customers", "what happened", "pattern", "sequence",
}

# --- VectorCypherRetriever config ---

RETRIEVAL_QUERY = """
MATCH (node)<-[:HAD_INTERACTION]-(cu:Customer)
MATCH (node)-[:DURING_CAMPAIGN]->(c:Campaign)-[:MANAGED_BY]->(a:Agency)
MATCH (node)-[:FOR_PRODUCT]->(p:Product)
OPTIONAL MATCH (node)-[:LINKED_TO_DEAL]->(d:Deal)
RETURN
    node.event_type + ' via ' + node.channel +
    ' | customer: ' + cu.customer_id +
    ' (' + cu.segment + ', ' + cu.region + ')' +
    ' | campaign: ' + c.campaign_id +
    ' | agency: ' + a.agency_id +
    ' | product: ' + p.product_id +
    ' | deal: ' + coalesce(d.deal_id, 'none') +
    ' | spend: $' + toString(node.spend_usd) +
    ' | revenue: $' + toString(node.revenue_usd)
    AS text,
    score
"""

# --- Text2CypherRetriever config ---

NEO4J_SCHEMA = """
Nodes:
  - Customer {customer_id, segment, region}
  - Interaction {interaction_id, occurred_at, channel, event_type, session_id, spend_usd, revenue_usd}
  - Campaign {campaign_id}
  - Product {product_id}
  - Agency {agency_id}
  - Creative {creative_id}
  - Deal {deal_id}
  - Session {session_id}

Relationships:
  (Customer)-[:HAD_INTERACTION]->(Interaction)
  (Interaction)-[:DURING_CAMPAIGN]->(Campaign)
  (Interaction)-[:FOR_PRODUCT]->(Product)
  (Interaction)-[:VIA_CREATIVE]->(Creative)
  (Campaign)-[:MANAGED_BY]->(Agency)
  (Customer)-[:HAS_DEAL]->(Deal)
  (Interaction)-[:IN_SESSION]->(Session)
"""

CYPHER_EXAMPLES = [
    "Which agencies drove the most revenue? => MATCH (a:Agency)<-[:MANAGED_BY]-(c:Campaign)<-[:DURING_CAMPAIGN]-(i:Interaction) WHERE i.revenue_usd > 0 RETURN a.agency_id, sum(i.revenue_usd) AS total_revenue ORDER BY total_revenue DESC LIMIT 5",
    "What channels are most effective at generating purchase events? => MATCH (i:Interaction) WHERE i.event_type = 'purchase' RETURN i.channel, count(i) AS purchase_count ORDER BY purchase_count DESC",
    "Which campaigns have the best return on spend? => MATCH (i:Interaction)-[:DURING_CAMPAIGN]->(c:Campaign) WHERE i.spend_usd > 0 RETURN c.campaign_id, sum(i.revenue_usd)/sum(i.spend_usd) AS roi ORDER BY roi DESC LIMIT 5",
    "Which products have the highest revenue in the APAC region? => MATCH (cu:Customer)-[:HAD_INTERACTION]->(i:Interaction)-[:FOR_PRODUCT]->(p:Product) WHERE cu.region = 'APAC' AND i.revenue_usd > 0 RETURN p.product_id, sum(i.revenue_usd) AS total_revenue ORDER BY total_revenue DESC LIMIT 5",
]

SAMPLE_QUESTIONS = [
    # Hybrid — needs both retrievers
    "Which agency drove the most revenue from customers who booked a demo?",
    "Which channels generated the most revenue from Enterprise customers, and what does their journey look like?",
    "Which products have the highest ROI and what type of customers typically purchase them?",
    # Pure contextual — VectorCypher only
    "What is the typical customer journey before a purchase event?",
    # Pure aggregate — Text2Cypher only
    "Which campaigns have the best return on spend?",
]


def classify_question(question: str) -> str:
    """
    Classify a question as 'contextual', 'aggregate', or 'hybrid'.
    Hybrid = has both aggregation signals AND contextual/relational signals.
    """
    q = question.lower()
    is_agg = any(kw in q for kw in AGGREGATION_KEYWORDS)
    is_ctx = any(kw in q for kw in CONTEXTUAL_KEYWORDS)

    if is_agg and is_ctx:
        return "hybrid"
    elif is_agg:
        return "aggregate"
    else:
        return "contextual"


def build_retrievers(driver):
    llm = OpenAILLM(model_name="gpt-4o-mini", model_params={"temperature": 0})
    embedder = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_retriever = VectorCypherRetriever(
        driver=driver,
        index_name="interaction_embeddings",
        embedder=embedder,
        retrieval_query=RETRIEVAL_QUERY,
        result_formatter=lambda r: RetrieverResultItem(content=r.get("text", "")),
    )
    cypher_retriever = Text2CypherRetriever(
        driver=driver,
        llm=llm,
        neo4j_schema=NEO4J_SCHEMA,
        examples=CYPHER_EXAMPLES,
    )

    vector_rag = GraphRAG(retriever=vector_retriever, llm=llm)
    cypher_rag = GraphRAG(retriever=cypher_retriever, llm=llm)

    return vector_rag, cypher_rag, llm


def search(vector_rag, cypher_rag, llm, question):
    """
    Route the question and return an answer.
    Hybrid questions run both retrievers and pass combined context to the LLM.
    """
    mode = classify_question(question)
    print(f"[mode: {mode}]")

    if mode == "contextual":
        return vector_rag.search(query_text=question, retriever_config={"top_k": 10})

    if mode == "aggregate":
        return cypher_rag.search(query_text=question)

    # Hybrid: run both retrievers, concatenate results, synthesize with LLM
    vector_result = vector_rag.search(query_text=question, retriever_config={"top_k": 10})
    cypher_result = cypher_rag.search(query_text=question)

    # Combine both answers into a single synthesis prompt
    combined_context = (
        f"--- Relational context (graph traversal) ---\n{vector_result.answer}\n\n"
        f"--- Aggregate context (full graph query) ---\n{cypher_result.answer}"
    )
    synthesis_prompt = (
        f"Using both the relational context and the aggregate data below, "
        f"provide a single comprehensive answer to: '{question}'\n\n{combined_context}"
    )

    # Use the LLM directly for final synthesis
    response = llm.invoke(synthesis_prompt)
    # Wrap in a simple object that matches what callers expect
    class Result:
        def __init__(self, answer):
            self.answer = answer
    return Result(response.content)


def main():
    driver = GraphDatabase.driver(
        NEO4J_URI.replace("neo4j+s://", "neo4j+ssc://"),
        auth=(NEO4J_USER, NEO4J_PASSWORD),
    )
    vector_rag, cypher_rag, llm = build_retrievers(driver)

    print("\nMarketing GraphRAG (Hybrid Fusion) — type a question or 'demo' to run sample questions.\n")

    while True:
        user_input = input("Question: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break
        if user_input.lower() == "demo":
            for q in SAMPLE_QUESTIONS:
                print(f"\nQ: {q}")
                result = search(vector_rag, cypher_rag, llm, q)
                print(f"A: {result.answer}\n")
        else:
            result = search(vector_rag, cypher_rag, llm, user_input)
            print(f"\nA: {result.answer}\n")

    driver.close()


if __name__ == "__main__":
    main()
