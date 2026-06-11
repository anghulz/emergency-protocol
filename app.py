import os
from pathlib import Path
import streamlit as st
from rag import DocumentIngestor, DocumentStore, RAGPipeline

CORPUS_DIR = Path(__file__).parent / "data" / "corpus"

st.set_page_config(
    page_title="Emergency Protocol",
    page_icon="🚨",
    layout="wide",
)

def get_api_key() -> str | None:
    """Read the OpenAI API key from Streamlit secrets or the environment."""
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except FileNotFoundError:
        pass  # no secrets.toml when running locally with env vars
    return os.getenv("OPENAI_API_KEY")


def init_session() -> bool:
    """Create the store/pipeline once per session. Returns False if no key."""
    api_key = get_api_key()
    if not api_key:
        return False

    if "store" not in st.session_state:
        st.session_state.store = DocumentStore(api_key=api_key)
        st.session_state.pipeline = RAGPipeline(
            store=st.session_state.store, api_key=api_key
        )
        st.session_state.ingestor = DocumentIngestor(chunk_size=1000, overlap=200)
        st.session_state.messages = []
        st.session_state.ingested_files = set()
    return True


def ingest_uploaded_files(uploaded_files) -> None:
    """Chunk, embed, and index files from the Streamlit uploader."""
    ingestor = st.session_state.ingestor
    store = st.session_state.store

    for uploaded in uploaded_files:
        if uploaded.name in st.session_state.ingested_files:
            continue  # already indexed this session
        try:
            with st.spinner(f"Indexing {uploaded.name}…"):
                chunks = ingestor.ingest(uploaded, filename=uploaded.name)
                added = store.add_chunks(chunks)
            st.success(f"Indexed **{uploaded.name}** ({added} chunks).")
            st.session_state.ingested_files.add(uploaded.name)
        except ValueError as exc:
            st.error(f"Could not ingest {uploaded.name}: {exc}")
        except Exception:
            st.error(
                f"Indexing failed for {uploaded.name}. "
                "Check the API key and try again."
            )


def ingest_corpus() -> None:
    """Index every supported file in the pre-loaded corpus directory."""
    files = [
        p for p in sorted(CORPUS_DIR.glob("*"))
        if p.suffix.lower() in DocumentIngestor.SUPPORTED_EXTENSIONS
    ]
    if not files:
        st.warning("No corpus files found in data/corpus/.")
        return

    for path in files:
        if path.name in st.session_state.ingested_files:
            continue
        try:
            with st.spinner(f"Indexing {path.name}…"):
                chunks = st.session_state.ingestor.ingest(path)
                st.session_state.store.add_chunks(chunks)
            st.session_state.ingested_files.add(path.name)
        except ValueError as exc:
            st.error(f"Could not ingest {path.name}: {exc}")
    st.success("Pre-loaded corpus indexed.")

def page_home() -> None:
    """Homepage: what the tool is, who it's for, how to use it."""
    st.title("🚨 Emergency Protocol")
    st.subheader("A protocol assistant for emergency management coordinators")

    st.markdown(
        """
**The problem.** Emergency coordinators work from hundreds of pages of dense
official guidance — FEMA frameworks, Red Cross shelter standards, state and
county emergency operations plans. During planning (or an actual activation),
nobody has time to page through six PDFs to find one standard.

**What this does.** Ask a question in plain language. Emergency Protocol
searches the loaded guidance documents, finds the relevant sections, and
answers **using only those documents** — with citations. If the documents
don't cover it, it says so instead of guessing.

**How to use it:**
1. Open **Document Library** and load the pre-loaded corpus (or upload your
   agency's own PDFs).
2. Open **Ask the Assistant** and ask your question.
3. Check the **References** under each answer to verify the source.

**Example questions:**
- *What public notification steps are recommended before a mandatory evacuation order?*
- *What are the space requirements per person in an emergency shelter?*
- *Who has the authority to activate the Emergency Operations Center?*
        """
    )
    st.info(
        "Emergency Protocol is a planning and training aid. During a real "
        "incident, always follow your agency's official procedures and chain "
        "of command."
    )


def page_documents() -> None:
    """Document management: upload, load corpus, view/remove indexed docs."""
    st.title("📚 Document Library")
    st.caption(
        "Documents indexed here become the assistant's only source of truth."
    )

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Load the pre-loaded corpus")
        st.markdown(
            "Official public guidance bundled with the app "
            "(FEMA, Red Cross, state plans)."
        )
        if st.button("Load pre-loaded corpus", type="primary"):
            ingest_corpus()

    with col_right:
        st.markdown("#### Upload your own documents")
        uploads = st.file_uploader(
            "PDF, TXT, or Markdown",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
        )
        if uploads:
            ingest_uploaded_files(uploads)

    st.divider()
    st.markdown("#### Currently indexed")

    store = st.session_state.store
    if store.chunk_count == 0:
        st.warning("No documents loaded yet. The assistant can't answer until "
                   "at least one document is indexed.")
        return

    for source, count in store.sources.items():
        col_a, col_b = st.columns([5, 1])
        col_a.markdown(f"**{source}** — {count} chunks")
        if col_b.button("Remove", key=f"rm_{source}"):
            store.remove_source(source)
            st.session_state.ingested_files.discard(source)
            st.rerun()


def page_chat() -> None:
    """Chat interface with response display and retrieved-chunk view."""
    st.title("💬 Ask the Assistant")

    store = st.session_state.store
    if store.chunk_count == 0:
        st.warning(
            "No documents are loaded. Go to the **Document Library** first."
        )
        return

    st.caption(
        f"Answering from {len(store.sources)} document(s), "
        f"{store.chunk_count} indexed chunks."
    )

    # Replay history.
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("chunks"):
                render_references(msg["chunks"])

    question = st.chat_input("Ask about protocols, standards, or procedures…")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the guidance documents…"):
            result = st.session_state.pipeline.answer(question)
        st.markdown(result["answer"])
        if result["chunks"]:
            render_references(result["chunks"])

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "chunks": result["chunks"]}
    )


def render_references(chunks: list[dict]) -> None:
    """Show which chunks were retrieved, with similarity scores."""
    with st.expander("📎 References — retrieved passages"):
        for i, chunk in enumerate(chunks, start=1):
            st.markdown(
                f"**{i}. {chunk['source']}** &nbsp; "
                f"(similarity {chunk['score']:.2f})"
            )
            st.caption(chunk["text"][:600] + ("…" if len(chunk["text"]) > 600 else ""))

def main() -> None:
    if not init_session():
        st.title("🚨 Emergency Protocol")
        st.error(
            "No OpenAI API key found. Set `OPENAI_API_KEY` as an environment "
            "variable or in `.streamlit/secrets.toml`, then reload."
        )
        st.stop()

    with st.sidebar:
        st.markdown("## 🚨 Emergency Protocol")
        page = st.radio(
            "Navigation", ["Home", "Document Library", "Ask the Assistant"],
            label_visibility="collapsed",
        )
        st.divider()
        store = st.session_state.store
        st.metric("Documents indexed", len(store.sources))
        st.metric("Chunks indexed", store.chunk_count)

    if page == "Home":
        page_home()
    elif page == "Document Library":
        page_documents()
    else:
        page_chat()


if __name__ == "__main__":
    main()
