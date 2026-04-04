"""
Generate OpenAI embeddings for Interaction nodes and store them on Neo4j.

Usage:
    python scripts/embed_interactions.py
"""

import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI

load_dotenv()

NEO4J_URI      = os.environ["NEO4J_URI"]
NEO4J_USER     = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# Model output size must match the vector index dimensions below (1536).
EMBEDDING_MODEL = "text-embedding-3-small"
# Chunk size for OpenAI embedding API calls (well under the per-request input limit).
BATCH_SIZE = 100


def fetch_interactions(session):
    # Run one Cypher query that walks from each Interaction to its related nodes.
    # Mandatory paths ensure we only return interactions that have campaign, customer, and product.
    # Optional deal link covers rows where there is no deal yet.
    # Returned columns become the fields used to build embeddable text.
    return session.run("""
        MATCH (i:Interaction)-[:DURING_CAMPAIGN]->(c:Campaign)-[:MANAGED_BY]->(a:Agency),
              (i)<-[:HAD_INTERACTION]-(cu:Customer),
              (i)-[:FOR_PRODUCT]->(p:Product)
        OPTIONAL MATCH (i)-[:LINKED_TO_DEAL]->(d:Deal)
        RETURN
            i.interaction_id  AS id,
            i.event_type      AS event_type,
            i.channel         AS channel,
            i.spend_usd       AS spend_usd,
            i.revenue_usd     AS revenue_usd,
            cu.segment        AS segment,
            cu.region         AS region,
            c.campaign_id     AS campaign,
            a.agency_id       AS agency,
            p.product_id      AS product,
            d.deal_id         AS deal
    """).data()


def build_text(record):
    # Append deal wording only when OPTIONAL MATCH found a deal; otherwise omit it.
    deal_part = f" linked to {record['deal']}" if record["deal"] else ""
    # Assemble one natural-language line so the embedding captures channel, segment, geography,
    # campaign, agency, optional deal, and money fields—not just a bare event type.
    return (
        f"{record['event_type']} interaction via {record['channel']} "
        f"for product {record['product']} by {record['segment']} customer "
        f"in {record['region']}, campaign {record['campaign']} "
        f"managed by {record['agency']}{deal_part}, "
        f"spend ${record['spend_usd'] or 0:.2f}, revenue ${record['revenue_usd'] or 0:.2f}"
    )


def create_vector_index(session):
    # Declare a cosine vector index on Interaction.embedding so later GraphRAG queries can do similarity search.
    # IF NOT EXISTS keeps reruns idempotent; dimensions must match EMBEDDING_MODEL output.
    session.run("""
        CREATE VECTOR INDEX interaction_embeddings IF NOT EXISTS
        FOR (i:Interaction) ON (i.embedding)
        OPTIONS {indexConfig: {
            `vector.dimensions`: 1536,
            `vector.similarity_function`: 'cosine'
        }}
    """)
    print("Vector index ready.")


def write_embeddings(session, records, embeddings):
    # Pair each graph row with its corresponding vector in the same order as the API returned them.
    rows = [{"id": r["id"], "embedding": e} for r, e in zip(records, embeddings)]
    # UNWIND turns that list into rows inside one transaction: match each Interaction by id and set embedding.
    session.run("""
        UNWIND $rows AS row
        MATCH (i:Interaction {interaction_id: row.id})
        SET i.embedding = row.embedding
    """, rows=rows)


def main():
    # Open a Bolt driver; neo4j+ssc avoids strict cert verification issues some macOS setups hit with neo4j+s.
    driver = GraphDatabase.driver(NEO4J_URI.replace("neo4j+s://", "neo4j+ssc://"), auth=(NEO4J_USER, NEO4J_PASSWORD))
    client = OpenAI(api_key=OPENAI_API_KEY)

    with driver.session() as session:
        # Ensure the vector index exists before we start writing embedding properties.
        create_vector_index(session)

        print("Fetching interactions from Neo4j...")
        records = fetch_interactions(session)
        print(f"Found {len(records)} interactions.")

        # Turn every DB row into the string that will be embedded (same order as records).
        texts = [build_text(r) for r in records]

        # Call OpenAI in slices: each call returns one vector per input string; accumulate in order.
        print("Generating embeddings via OpenAI...")
        all_embeddings = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
            all_embeddings.extend([item.embedding for item in response.data])
            print(f"  Embedded {min(i + BATCH_SIZE, len(texts))}/{len(texts)}")

        # Persist vectors back to Neo4j in the same batching pattern to limit payload size per query.
        print("Writing embeddings back to Neo4j...")
        for i in range(0, len(records), BATCH_SIZE):
            write_embeddings(
                session,
                records[i : i + BATCH_SIZE],
                all_embeddings[i : i + BATCH_SIZE],
            )

    driver.close()
    print("Done. All interactions now have embeddings.")


if __name__ == "__main__":
    main()
