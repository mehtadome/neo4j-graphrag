"""
GraphRAG app using VectorCypherRetriever.
Best for contextual and relational questions — finds semantically similar Interaction
nodes via vector search, then traverses the graph to pull in connected context.

Usage:
    python scripts/vector_app.py

How RAG_OBJECT.search() works:
    * Embedder calls OpenAI with questions string and returns a vector embedding.
    * Vector is sent to Neo4j's interaction_embeddings index for similarity search.
    * Neo4j returns the top 10 most semantically similar interactions.
    * For each of the 10 matched nodes, runs RETRIEVAL_QUERY to walk the path.
    * Concantenate all 10 strings into a context block using a default prompt template, along with the user's question.
    * Call OpenAI with the context block and question to generate an answer.
    * Return the answer.
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

# Each node is a vector-matched Interaction node = interactions whose embedded text was most semantically similar to the query.
RETRIEVAL_QUERY = """
# Match interaction to customer to campaign to agency to product to deal
MATCH (node)<-[:HAD_INTERACTION]-(cu:Customer)
MATCH (node)-[:DURING_CAMPAIGN]->(c:Campaign)-[:MANAGED_BY]->(a:Agency)
MATCH (node)-[:FOR_PRODUCT]->(p:Product)

# Check if the interaction is linked to a deal
OPTIONAL MATCH (node)-[:LINKED_TO_DEAL]->(d:Deal)

# Return the text of the interaction and the score
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
    # Temperature 0 means deterministic output
    llm = OpenAILLM(model_name="gpt-4o-mini", model_params={"temperature": 0})
    # create VectorCypherRetriever object
    retriever = VectorCypherRetriever(
        driver=driver,
        index_name="interaction_embeddings", # name of the vector index we created in embed_interactions.py
        embedder=embedder,
        retrieval_query=RETRIEVAL_QUERY,
        # Format conversation
        result_formatter=lambda r: RetrieverResultItem(content=r.get("text", "")),
    )
    # create GraphRAG object
    return GraphRAG(retriever=retriever, llm=llm)


def main():
    driver = GraphDatabase.driver(
        NEO4J_URI.replace("neo4j+s://", "neo4j+ssc://"),
        auth=(NEO4J_USER, NEO4J_PASSWORD),
    )
    # rag is the GraphRAG object imbued with the VectorCypherRetriever object
    rag = build_rag(driver)

    print("\nMarketing GraphRAG (VectorCypher) — type a question or 'demo' to run sample questions.\n")

    while True:
        user_input = input("Question: ").strip()
        # Continue if user types nothing
        if not user_input:
            continue
        # Exit if user types exit or quit
        if user_input.lower() in ("exit", "quit"):
            break
        # Run sample questions if user types demo
        if user_input.lower() == "demo":
            for q in SAMPLE_QUESTIONS:
                print(f"\nQ: {q}")
                # Set top_k to 10 to get the top 10 most relevant interactions
                result = rag.search(query_text=q, retriever_config={"top_k": 10})
                print(f"\n\n{'='*60}\n{result.answer}\n{'='*60}\n")

        # Run user question if user types anything else
        else:
            result = rag.search(query_text=user_input, retriever_config={"top_k": 10})
            print(f"\n\n{'='*60}\n{result.answer}\n{'='*60}\n")

    driver.close()


if __name__ == "__main__":
    main()
