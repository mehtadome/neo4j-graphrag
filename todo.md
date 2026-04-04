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
- [ ] Merge vector and text2cypher apps into a single `graphrag_app.py` with auto-routing
