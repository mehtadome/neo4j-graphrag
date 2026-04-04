"""
Step 1: Generate OpenAI embeddings for Interaction nodes and write them back to Neo4j.
Run this once before starting the GraphRAG app.

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

EMBEDDING_MODEL = "text-embedding-3-small"  # 1536 dimensions, cheapest OpenAI embeddings
BATCH_SIZE = 100  # OpenAI supports up to 2048 inputs per request


def fetch_interactions(session):
    # Traverse the graph to collect each Interaction alongside its related context.
    # We join to Campaign, Agency, Customer, and Product via mandatory relationships,
    # then optionally join to Deal (not all interactions are linked to a deal).
    # This multi-hop traversal is the core of GraphRAG — we're pulling relational
    # context that would be impossible to get from a flat document embedding alone.
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
    # Convert a graph record into a single descriptive sentence.
    # This sentence is what actually gets embedded — the richer and more specific it is,
    # the better the vector search will perform at query time.
    # Fields like segment, region, agency, and deal give the embedding relational meaning
    # that a raw event log row would not have on its own.
    deal_part = f" linked to {record['deal']}" if record["deal"] else ""
    return (
        f"{record['event_type']} interaction via {record['channel']} "
        f"for product {record['product']} by {record['segment']} customer "
        f"in {record['region']}, campaign {record['campaign']} "
        f"managed by {record['agency']}{deal_part}, "
        f"spend ${record['spend_usd'] or 0:.2f}, revenue ${record['revenue_usd'] or 0:.2f}"
    )


def create_vector_index(session):
    # Create a vector index on Interaction.embedding if one doesn't already exist.
    # The index dimensions must exactly match the embedding model output (1536 for
    # text-embedding-3-small). Cosine similarity is standard for semantic search.
    # IF NOT EXISTS makes this safe to re-run without error.
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
    # Write each embedding vector back to its Interaction node as the `embedding` property.
    # UNWIND lets us send a full batch in one Bolt round-trip rather than one query per node,
    # which is significantly faster at scale. We match by interaction_id to ensure we're
    # setting the right embedding on the right node.
    session.run("""
        UNWIND $rows AS row
        MATCH (i:Interaction {interaction_id: row.id})
        SET i.embedding = row.embedding
    """, rows=[{"id": r["id"], "embedding": e} for r, e in zip(records, embeddings)])


def main():
    # neo4j+s:// enforces TLS with certificate verification. We swap to neo4j+ssc://
    # (self-signed certificate) to bypass the macOS SSL chain verification issue
    # without disabling encryption entirely.
    driver = GraphDatabase.driver(NEO4J_URI.replace("neo4j+s://", "neo4j+ssc://"), auth=(NEO4J_USER, NEO4J_PASSWORD))
    client = OpenAI(api_key=OPENAI_API_KEY)

    with driver.session() as session:
        create_vector_index(session)

        print("Fetching interactions from Neo4j...")
        records = fetch_interactions(session)
        print(f"Found {len(records)} interactions.")

        # Build the text representation of each interaction before hitting the OpenAI API
        texts = [build_text(r) for r in records]

        # Send texts to OpenAI in batches to generate embedding vectors.
        # Each vector is a list of 1536 floats representing the semantic meaning of the text.
        print("Generating embeddings via OpenAI...")
        all_embeddings = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
            all_embeddings.extend([item.embedding for item in response.data])
            print(f"  Embedded {min(i + BATCH_SIZE, len(texts))}/{len(texts)}")

        # Write embeddings back to Neo4j in batches matching the fetch order
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
