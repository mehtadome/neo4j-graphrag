"""
GraphRAG app using Text2CypherRetriever.
Best for analytical and aggregation questions — the LLM generates a Cypher query
that runs against the full graph, rather than searching a top-K slice.

Usage:
    python scripts/text2cypher_app.py

Syntax Note:
    * (a)<-[:REL]-(b) — arrow points left, relationship goes from b to a
    * (a)-[:REL]->(b) — arrow points right, relationship goes from a to b

How RAG_OBJECT.search() works:
    * Creates the appropriate objects for the retriever and generator.
    * Assemble a prompt template using NEO4J_SCHEMA + EXAMPLES + the question.
    * OpenAI reads prompt and writes a Cypher query, using the schema and examples to guide it.
    * Cypher query is run against the full graph, not a top-K slice.
    * Raw results are fed back to OpenAI to generate an answer.
    * Return the answer.
"""

import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import Text2CypherRetriever
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.generation import GraphRAG

load_dotenv()

NEO4J_URI      = os.environ["NEO4J_URI"]
NEO4J_USER     = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

# Graph schema provided to the LLM so it knows what nodes and relationships exist
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

# Few-shot examples to guide the LLM toward correct Cypher for our schema
EXAMPLES = [
    "Which agencies drove the most revenue? => MATCH (a:Agency)<-[:MANAGED_BY]-(c:Campaign)<-[:DURING_CAMPAIGN]-(i:Interaction) WHERE i.revenue_usd > 0 RETURN a.agency_id, sum(i.revenue_usd) AS total_revenue ORDER BY total_revenue DESC LIMIT 5",
    "What channels are most effective at generating purchase events? => MATCH (i:Interaction) WHERE i.event_type = 'purchase' RETURN i.channel, count(i) AS purchase_count ORDER BY purchase_count DESC",
    "Which campaigns have the best return on spend? => MATCH (i:Interaction)-[:DURING_CAMPAIGN]->(c:Campaign) WHERE i.spend_usd > 0 RETURN c.campaign_id, sum(i.revenue_usd)/sum(i.spend_usd) AS roi ORDER BY roi DESC LIMIT 5",
    "Which products have the highest revenue in the APAC region? => MATCH (cu:Customer)-[:HAD_INTERACTION]->(i:Interaction)-[:FOR_PRODUCT]->(p:Product) WHERE cu.region = 'APAC' AND i.revenue_usd > 0 RETURN p.product_id, sum(i.revenue_usd) AS total_revenue ORDER BY total_revenue DESC LIMIT 5",
]
# Example 1:
# Agency that managed a Campaign during which interactions occured that generated revenue

# Example 2:
# Interaction where event type is a purchase

# Example 3:
# Interaction during a campaign that generated revenue

# Example 4:
# Customers in the APAC region who had an interaction with a product that generated revenue

SAMPLE_QUESTIONS = [
    "Which agencies drove the most revenue from Enterprise customers?",
    "Which campaigns have the best return on spend?",
    "Which products have the highest revenue in the APAC region?",
    "What channels are most effective at generating purchase events?",
]


def build_rag(driver):
    # Temperature 0 means deterministic output
    llm = OpenAILLM(model_name="gpt-4o-mini", model_params={"temperature": 0})
    # create Text2CypherRetriever object
    retriever = Text2CypherRetriever(
        driver=driver,
        llm=llm,
        # Graph schema provided to the LLM so it knows what nodes and relationships exist
        neo4j_schema=NEO4J_SCHEMA,
        # Few-shot examples to guide the LLM toward correct Cypher for our schema
        examples=EXAMPLES,
    )
    # create GraphRAG object
    return GraphRAG(retriever=retriever, llm=llm)


def main():
    # Bolt connection to Aura; neo4j+ssc matches the embed script when strict TLS verification is problematic.
    driver = GraphDatabase.driver(
        NEO4J_URI.replace("neo4j+s://", "neo4j+ssc://"),
        auth=(NEO4J_USER, NEO4J_PASSWORD),
    )
    # rag is the GraphRAG object imbued with the Text2CypherRetriever object
    rag = build_rag(driver)

    print("\nMarketing GraphRAG (Text2Cypher) — type a question or 'demo' to run sample questions.\n")

    while True:
        user_input = input("Question: ").strip()
        # Ignore blank lines so accidental Enter does not trigger a search.
        if not user_input:
            continue
        # Exit if user types exit or quit
        if user_input.lower() in ("exit", "quit"):
            break
        # Run sample questions if user types demo
        if user_input.lower() == "demo":
            for q in SAMPLE_QUESTIONS:
                print(f"\nQ: {q}")
                # Same search settings as a normal question (see branch below for retriever_config).
                result = rag.search(query_text=q)
                print(f"A: {result.answer}\n")

        # Run user question if user types anything else
        else:
            result = rag.search(query_text=user_input)
            print(f"\nA: {result.answer}\n")

    driver.close()


if __name__ == "__main__":
    main()
