"""
Main GraphRAG app with auto-routing between VectorCypherRetriever and Text2CypherRetriever.
Questions are automatically routed based on whether they are contextual/relational
or analytical/aggregation.

For isolated retriever testing use:
  - vector_app.py     (VectorCypherRetriever only)
  - text2cypher_app.py (Text2CypherRetriever only)

Usage:
    python scripts/graphrag_app.py
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

# Keywords that signal an aggregation question — route to Text2CypherRetriever
AGGREGATION_KEYWORDS = {
    "most", "best", "highest", "lowest", "total", "count",
    "average", "top", "least", "rank", "how many", "sum",
    "which campaign", "which channel", "which product", "which agency",
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

# --- Sample questions spanning both retriever types ---

SAMPLE_QUESTIONS = [
    "What is the typical customer journey before a purchase event?",
    "What products did customers who booked a demo end up purchasing?",
    "Which agencies drove the most revenue from Enterprise customers?",
    "Which campaigns have the best return on spend?",
    "Which products have the highest revenue in the APAC region?",
]


def is_aggregation(question: str) -> bool:
    """Return True if the question looks analytical — route to Text2CypherRetriever."""
    q = question.lower()
    return any(kw in q for kw in AGGREGATION_KEYWORDS)


def build_rags(driver):
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

    return GraphRAG(retriever=vector_retriever, llm=llm), GraphRAG(retriever=cypher_retriever, llm=llm)


def search(rag_vector, rag_cypher, question):
    if is_aggregation(question):
        mode, rag = "Text2Cypher", rag_cypher
        result = rag.search(query_text=question)
    else:
        mode, rag = "VectorCypher", rag_vector
        result = rag.search(query_text=question, retriever_config={"top_k": 10})
    print(f"[retriever: {mode}]")
    return result


def main():
    driver = GraphDatabase.driver(
        NEO4J_URI.replace("neo4j+s://", "neo4j+ssc://"),
        auth=(NEO4J_USER, NEO4J_PASSWORD),
    )
    rag_vector, rag_cypher = build_rags(driver)

    print("\nMarketing GraphRAG — type a question or 'demo' to run sample questions.\n")

    while True:
        user_input = input("Question: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break
        if user_input.lower() == "demo":
            for q in SAMPLE_QUESTIONS:
                print(f"\nQ: {q}")
                result = search(rag_vector, rag_cypher, q)
                print(f"A: {result.answer}\n")
        else:
            result = search(rag_vector, rag_cypher, user_input)
            print(f"\nA: {result.answer}\n")

    driver.close()


if __name__ == "__main__":
    main()
