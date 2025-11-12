# Science RAG System

A Retrieval-Augmented Generation (RAG) system designed for multi-chapter science textbooks with nested subtopics. It uses a local LLM (via Ollama) and a vector database (ChromaDB) to provide context-aware answers, with special support for filtering questions by assigned marks.
# Installation
ollama pull llama3.2
pip install ollama chromadb sentence-transformers
python rag_system.py chemistry_data.json


## Features

* ✅ **Vector-based Retrieval**: Fast semantic search using `sentence-transformers`.
* ✅ **Context-aware Answers**: Uses relevant textbook content pulled from a vector store.
* ✅ **Mark-specific Filtering**: Filter questions by marks (e.g., "1 marks", "3 marks").
* ✅ **Interactive CLI**: Easy-to-use command-line interface.
* ✅ **Local First**: Runs entirely locally with Ollama.
* ✅ **Metadata Tracking**: Understands chapter, topic, and subtopic context.
* ✅ **Flexible Queries**: Ask questions in natural language.

---

## How It Works

1.  **Data Loading**: Reads and parses your structured JSON textbook data (key concepts, questions, answers).
2.  **Embedding Creation**: Converts all text chunks (concepts, Q&As) into vector embeddings using `all-MiniLM-L6-v2`.
3.  **Vector Storage**: Stores these embeddings and their associated metadata (chapter, topic, marks) in a ChromaDB collection.
4.  **Query Processing**:
    * The user asks a question (e.g., "What is oxidation [3 marks]").
    * The system embeds the query and performs a similarity search in ChromaDB.
    * It filters results based on any specified "marks" metadata.
5.  **Answer Generation**: The retrieved context (the most relevant textbook snippets) is passed to an Ollama LLM (e.g., `llama3.2`) along with the original question to synthesize a final answer.

---

## Prerequisites

1.  **Python 3.8+**
    * Ensure Python is installed on your system.

2.  **Ollama**
    * **Install Ollama**: Visit [https://ollama.ai/download](https://ollama.ai/download) and install it for your OS.
    * **Pull a Model**: You need at least one model. `llama3.2` is a good default.
        ```bash
        ollama pull llama3.2
        ```
    * **Ensure Ollama is Running**: The Ollama application or `ollama serve` command must be running in the background.

---

## Installation

1.  **Install Python Dependencies**
    ```bash
    pip install ollama chromadb sentence-transformers
    ```

2.  **Save Your Dataset**
    * Ensure your textbook data is formatted as JSON (see [Data Format](#data-format) below) and save it as a file (e.g., `chemistry_data.json`).

---

## Data Format

Your JSON data must follow this nested structure:

```json
[
  {
    "chapter_number": 1,
    "chapter_name": "Chemical Reactions and Equations",
    "topics": [
      {
        "topic_name": "Types of Chemical Reactions",
        "key_concepts": [
          "A combination reaction is a reaction in which two or more reactants combine to form a single product.",
          "Example: CaO(s) + H2O(l) -> Ca(OH)2(aq)"
        ],
        "questions": [
          {
            "one_mark_questions": [
              {
                "question": "What is a combination reaction?",
                "answer": "A reaction in which two or more reactants combine to form a single product."
              }
            ],
            "two_mark_questions": [
              {
                "question": "Why is respiration considered an exothermic reaction?",
                "answer": "Respiration is considered an exothermic reaction because energy is released during this process..."
              }
            ],
            "three_mark_questions": [],
            "five_mark_questions": []
          }
        ]
      }
    ]
  }
]