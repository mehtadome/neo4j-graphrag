# Scripts

## `generate_marketing_sample.py`
**One-time.** Generates `data/marketing_interactions.csv` with 500 rows of synthetic marketing interaction data. Seed is fixed at 42 for reproducibility. Only re-run if you need to regenerate the dataset from scratch.

## `patch_spend_revenue.py`
**One-time.** Patches `spend_usd` and `revenue_usd` onto `Interaction` nodes in Neo4j. Needed because the Aura Data Importer silently skipped these columns during the initial graph import — they exist in the CSV but were never mapped to node properties.

## `embed_interactions.py`
**One-time.** Fetches all `Interaction` nodes from Neo4j, builds a descriptive sentence for each by traversing to related `Customer`, `Campaign`, `Agency`, `Product`, and `Deal` nodes, then calls OpenAI `text-embedding-3-small` to generate 1536-dim vectors. Writes embeddings back to `Interaction.embedding` and creates the `interaction_embeddings` vector index. Must be run after `patch_spend_revenue.py` and before either app.

## `graphrag_app.py` — MAIN APP
**Ongoing.** The primary entry point. Auto-routes each question to the correct retriever based on keyword detection — `Text2CypherRetriever` for aggregation questions, `VectorCypherRetriever` for contextual/relational ones. Prints `[retriever: ...]` on each response so the routing decision is visible. Type `demo` to run all 5 sample questions, or `exit` to quit.

## `vector_app.py`
**Ongoing (isolated testing).** Runs only `VectorCypherRetriever`. Use this to test or demo the vector search + graph traversal path in isolation, without the auto-router.

## `text2cypher_app.py`
**Ongoing (isolated testing).** Runs only `Text2CypherRetriever`. Use this to test or demo the LLM-generated Cypher path in isolation, without the auto-router.
