
1. The Streamlit UI (`app.py`) accepts document uploads and user questions.
2. `DocumentIngestor` (`rag/ingestion.py`) parses PDF/TXT/MD files, cleans the text, and splits it into 1000-character chunks with 200-character overlap.
3. `DocumentStore` (`rag/store.py`) embeds each chunk with the OpenAI `text-embedding-3-small` model and indexes the vectors in an in-memory ChromaDB collection.
4. On each question, the store retrieves the top 4 chunks by cosine similarity.
5. `RAGPipeline` (`rag/pipeline.py`) assembles the retrieved chunks into a context string and sends it with the question to the OpenAI Chat API (`gpt-4o-mini`).
6. The prompt template (`rag/prompts.py`) enforces grounded answers, source citations, and an explicit "I don't have enough information" response when the context is insufficient.

## Run locally

1. Clone the repository and enter the project directory.
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create `.streamlit/secrets.toml` containing:
   ```toml
   OPENAI_API_KEY = "sk-your-key-here"
   ```
5. Start the app:
   ```bash
   streamlit run app.py
   ```
6. Open http://localhost:8501. In **Document Library**, load the pre-loaded corpus or upload documents, then ask questions in **Ask the Assistant**.

## Deploy to Streamlit Community Cloud

1. Push the repository to GitHub. The `.gitignore` excludes `secrets.toml`; verify no API key is committed.
2. Go to https://share.streamlit.io and select **New app**.
3. Choose the repository, branch `main`, and main file `app.py`.
4. Under **Advanced settings**, in the **Secrets** field, add:
   ```toml
   OPENAI_API_KEY = "sk-your-key-here"
   ```
5. Deploy, then record the app URL in this README and in the report.

```python
__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
```

## Project structure

```
emergency-protocol/
├── app.py                 # Streamlit UI
├── rag/
│   ├── ingestion.py       # DocumentIngestor: parse, clean, chunk
│   ├── store.py           # DocumentStore: embeddings, ChromaDB index
│   ├── pipeline.py        # RAGPipeline: retrieval, context assembly, LLM call
│   └── prompts.py         # Prompt templates
├── data/corpus/           # Pre-loaded guidance documents
├── .streamlit/config.toml # Theme configuration
└── requirements.txt
```

## Corpus

Place text-based PDF, TXT, or MD files in `data/corpus/`