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
- [ ] Research `HybridRetriever` in `neo4j-graphrag` and understand result merging
- [ ] Create `scripts/hybrid_app.py` — runs both retrievers in parallel and synthesizes a combined answer
- [ ] Test with hybrid questions (e.g. "Which agency drove the most revenue from customers who booked a demo?")
- [ ] Document tradeoffs (latency, API cost, answer quality) in `interview.md`

## Aura Agent (console UI)
- [ ] Open Neo4j Aura console and navigate to the Agent builder
- [ ] Connect agent to the existing graph (instance `1e4a65b0`)
- [ ] Verify the agent can see the vector index `interaction_embeddings`
- [ ] Test with contextual questions (customer journey, demo → purchase)
- [ ] Test with aggregation questions (ROI by campaign, revenue by agency)
- [ ] Note where it succeeds vs. where the custom pipeline does better
- [ ] Document differences in `interview.md`
