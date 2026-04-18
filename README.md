# Educational Chatbot

Educational Chatbot is a local-first science question-answering system that combines:

1. Retrieval-Augmented Generation (RAG) over textbook data using ChromaDB.
2. Optional web interface for interactive question answering.
3. Diagram/image mapping based on semantic relevance and OCR text matching.

## Documentation

Read the docs in this order:

1. [Web Interface Guide](docs/WEB_INTERFACE_README.md)
2. [Image Matching Logic](docs/IMAGE_MATCHING_LOGIC.md)

## Repository Layout

```text
Educational_Chatbot/
|-- app.py
|-- combined_qa_system.py
|-- final.py
|-- embeddings.pt
|-- science_db/
|-- text/
|   |-- rag_system.py
|   |-- syllabus_map_qna.json
|   |-- syllabus_map_key_concepts.json
|-- docs/
|   |-- WEB_INTERFACE_README.md
|   `-- IMAGE_MATCHING_LOGIC.md
`-- README.md
```

## Core Capabilities

1. Semantic retrieval from chapter/topic/subtopic science content.
2. Mark-aware responses when queries include values like `[2 marks]`.
3. Persistent vector database in `science_db/` for faster startup after indexing.
4. Diagram matching and OCR-driven label highlighting.

## Requirements

1. Python 3.8 or newer.
2. Ollama installed and running.
3. Ollama model pulled locally (default: `llama3.2`).

## Setup

```bash
ollama pull llama3.2
pip install ollama chromadb sentence-transformers
```

For web interface usage, also install Flask (or project-specific web dependencies).

## Run

### CLI RAG mode

```bash
python text/rag_system.py text/syllabus_map_qna.json
```

### Web app mode

```bash
python app.py
```

Then open `http://localhost:5000`.

## Data Notes

1. Textbook JSON data is expected under `text/`.
2. Embeddings and image assets are loaded from local files (`embeddings.pt`, image folders).
3. ChromaDB persistence path defaults to `./science_db`.

## Maintenance

1. Keep high-level project usage in this `README.md`.
2. Keep implementation details in `docs/`.
3. Avoid duplicating setup instructions across multiple files.