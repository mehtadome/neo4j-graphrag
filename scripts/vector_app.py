"""
GraphRAG app using VectorCypherRetriever.
Best for contextual and relational questions — finds semantically similar Interaction
nodes via vector search, then traverses the graph to pull in connected context.

Usage:
    python scripts/vector_app.py
"""

import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.types import RetrieverResultItem

load_dotenv()

NEO4J_URI      = os.environ["NEO4J_URI"]
NEO4J_USER     = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

# After each vector-matched Interaction node, traverse the graph to pull in
# connected context (Customer, Campaign, Agency, Product, Deal).
# This is what makes it GraphRAG rather than plain vector RAG.
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

SAMPLE_QUESTIONS = [
    "What is the typical customer journey before a purchase event?",
    "What products did customers who booked a demo end up purchasing?",
    "Which agencies drove the most revenue from Enterprise customers?",
]


def build_rag(driver):
    embedder = OpenAIEmbeddings(model="text-embedding-3-small")
    llm = OpenAILLM(model_name="gpt-4o-mini", model_params={"temperature": 0})
    retriever = VectorCypherRetriever(
        driver=driver,
        index_name="interaction_embeddings",
        embedder=embedder,
        retrieval_query=RETRIEVAL_QUERY,
        result_formatter=lambda r: RetrieverResultItem(content=r.get("text", "")),
    )
    return GraphRAG(retriever=retriever, llm=llm)


def main():
    driver = GraphDatabase.driver(
        NEO4J_URI.replace("neo4j+s://", "neo4j+ssc://"),
        auth=(NEO4J_USER, NEO4J_PASSWORD),
    )
    rag = build_rag(driver)

    print("\nMarketing GraphRAG (VectorCypher) — type a question or 'demo' to run sample questions.\n")

    while True:
        user_input = input("Question: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break
        if user_input.lower() == "demo":
            for q in SAMPLE_QUESTIONS:
                print(f"\nQ: {q}")
                result = rag.search(query_text=q, retriever_config={"top_k": 10})
                print(f"A: {result.answer}\n")
        else:
            result = rag.search(query_text=user_input, retriever_config={"top_k": 10})
            print(f"\nA: {result.answer}\n")

    driver.close()


if __name__ == "__main__":
    main()
