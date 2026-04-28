"""
Patches missing spend_usd and revenue_usd properties onto Interaction nodes.
Run once if these fields were not mapped during the initial Aura Data Importer run.

Usage:
    python scripts/patch_spend_revenue.py
"""

import csv
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

NEO4J_URI      = os.environ["NEO4J_URI"]
NEO4J_USER     = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

CSV_PATH = "data/marketing_interactions.csv"
BATCH_SIZE = 100


def main():
    rows = []
    with open(CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            # Match id to interaction_id, then cast spend_usd and revenue_usd to floats
            rows.append({
                "id":          row["interaction_id"],
                "spend_usd":   float(row["spend_usd"]),
                "revenue_usd": float(row["revenue_usd"]),
            })

    driver = GraphDatabase.driver(
        NEO4J_URI.replace("neo4j+s://", "neo4j+ssc://"),
        auth=(NEO4J_USER, NEO4J_PASSWORD),
    )

    with driver.session() as session:
        # Loop through rows in batches of BATCH_SIZE (currently 100)
        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i : i + BATCH_SIZE]
            session.run("""
                # Expand list so one transaction handles 100 nodes at a time
                UNWIND $rows AS row
                # Find existing node
                MATCH (i:Interaction {interaction_id: row.id})
                # Write in new properties
                SET i.spend_usd   = row.spend_usd,
                    i.revenue_usd = row.revenue_usd
            """, rows=batch)
            # Print progress
            print(f"Patched {min(i + BATCH_SIZE, len(rows))}/{len(rows)}")

    driver.close()
    print("Done.")


if __name__ == "__main__":
    main()
