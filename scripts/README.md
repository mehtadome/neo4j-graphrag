# Scripts

## `generate_marketing_sample.py`
**One-time.** Generates `data/marketing_interactions.csv` with 500 rows of synthetic marketing interaction data. Seed is fixed at 42 for reproducibility. Only re-run if you need to regenerate the dataset from scratch.

## `patch_spend_revenue.py`
**One-time.** Patches `spend_usd` and `revenue_usd` onto `Interaction` nodes in Neo4j. Needed because the Aura Data Importer silently skipped these columns during the initial graph import — they exist in the CSV but were never mapped to node properties.

## `embed_interactions.py`
**One-time.** Fetches all `Interaction` nodes from Neo4j, builds a descriptive sentence for each by traversing to related `Customer`, `Campaign`, `Agency`, `Product`, and `Deal` nodes, then calls OpenAI `text-embedding-3-small` to generate 1536-dim vectors. Writes embeddings back to `Interaction.embedding` and creates the `interaction_embeddings` vector index. Must be run after `patch_spend_revenue.py` and before `graphrag_app.py`.

## `graphrag_app.py`
**Ongoing.** Interactive GraphRAG app. Accepts natural language questions, embeds them, finds semantically similar `Interaction` nodes via the vector index, traverses the graph for relational context, and answers via `gpt-4o-mini`. Type `demo` to run 5 sample questions, or `exit` to quit.
