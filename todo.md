# Project TODO

## Setup
- [x] Scaffold Python project with venv and requirements.txt
- [x] Generate synthetic marketing dataset (`scripts/generate_marketing_sample.py`)
- [x] Import CSV into Neo4j Aura via Data Importer (783 nodes, 2,666 relationships)
- [x] Add Session nodes post-import via Cypher
- [x] Create vector index on `Interaction.embedding`
- [x] Patch missing `spend_usd` / `revenue_usd` properties (`scripts/patch_spend_revenue.py`)
- [x] Generate and store OpenAI embeddings (`scripts/embed_interactions.py`)

## Demo
- [x] Run GraphRAG app and verify `demo` mode returns sensible answers
- [x] Confirm all 5 sample questions produce coherent, graph-grounded responses

## Text2Cypher
- [x] Run `text2cypher_app.py` and verify aggregation questions return correct results
- [x] Confirm `vector_app.py` and `text2cypher_app.py` work independently

## Final
- [x] Merge vector and text2cypher apps into a single `graphrag_app.py` with auto-routing

## Hybrid Retrieval
- [x] Research `HybridRetriever` in `neo4j-graphrag` and understand result merging
- [x] Create `scripts/hybrid_app.py` — runs both retrievers in parallel and synthesizes a combined answer
- [x] Test with hybrid questions (e.g. "Which agency drove the most revenue from customers who booked a demo?")
- [x] Document tradeoffs (latency, API cost, answer quality)

## Aura Agent (console UI)
- [x] Open Neo4j Aura console and navigate to the Agent builder
- [x] Connect agent to the existing graph (instance `1e4a65b0`)
- [x] Verify the agent can see the vector index `interaction_embeddings`
- [x] Test with contextual questions (customer journey, demo → purchase)
- [x] Test with aggregation questions (ROI by campaign, revenue by agency)
- [x] Note where it succeeds vs. where the custom pipeline does better
- [x] Document differences between the local agent and Aura agent
