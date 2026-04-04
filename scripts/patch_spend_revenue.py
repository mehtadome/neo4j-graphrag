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
        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i : i + BATCH_SIZE]
            session.run("""
                UNWIND $rows AS row
                MATCH (i:Interaction {interaction_id: row.id})
                SET i.spend_usd   = row.spend_usd,
                    i.revenue_usd = row.revenue_usd
            """, rows=batch)
            print(f"Patched {min(i + BATCH_SIZE, len(rows))}/{len(rows)}")

    driver.close()
    print("Done.")


if __name__ == "__main__":
    main()
