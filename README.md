# GraphRAG Agent - Local and Aura

Marketing attribution GraphRAG demo built on Neo4j Aura. The project was built in two phases: first a custom local Python agent with full control over retrieval and routing, then a no-code Aura Console Agent layered on the same graph.

## Architecture

There are two interfaces to the same Neo4j graph:

1. **Local Python Agent** (`scripts/graphrag_app.py`) — custom retrieval pipeline with three-way routing between vector, graph-traversal, and Text2Cypher retrievers
2. **Aura Console Agent** — Neo4j's built-in no-code agent UI, connected to the same graph and vector index

Building the local agent first gave a precise understanding of what the graph contains and how queries should be routed, which made configuring the Aura agent much more deliberate.

## Graph Model

![Graph Model](assets/graph-model.png)

## Setup

A virtual environment lives in `.venv`. The workspace is configured to use it as the default Python interpreter in Cursor/VS Code.

Activate it:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your credentials:

```
NEO4J_URI=neo4j+s://<your-aura-id>.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<your-password>
OPENAI_API_KEY=sk-...
```

## Run Order

```bash
python scripts/patch_spend_revenue.py   # one-time: patch missing CSV fields
python scripts/embed_interactions.py    # one-time: generate and store embeddings
python scripts/graphrag_app.py          # main app with auto-routing
```

For isolated retriever testing:

```bash
python scripts/vector_app.py            # VectorCypherRetriever only
python scripts/text2cypher_app.py       # Text2CypherRetriever only
```

## Example Responses

See [examples.md](examples.md) for live responses from the graph across all five sample questions, including which retriever was used for each.

## Requirements

- Python 3.10+
- Neo4j Aura instance with the marketing interactions graph loaded — the instance must be running before starting any script
- OpenAI API key (embeddings + GPT-4o-mini)
