import requests
import streamlit as st

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Financial Due Diligence RAG",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Financial Due Diligence RAG")
st.caption("Hybrid Search (Nomic + BM25) • Cross-Encoder Reranking • Grounded Citations")


def fetch_health() -> dict | None:
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def fetch_documents() -> list[str]:
    try:
        resp = requests.get(f"{API_BASE_URL}/documents", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return []


# --- Sidebar: System Status, PDF Ingestion & Query Filters ---
with st.sidebar:
    st.header("System & Filings")

    health = fetch_health()
    if health and health.get("qdrant_connected"):
        st.success(
            f"Qdrant Online • {health['indexed_points']} chunks in `{health['collection_name']}`"
        )
    else:
        st.error("FastAPI / Qdrant Offline. Start `uvicorn api.main:app` first.")

    st.divider()
    st.subheader("1. Upload Company Filing (PDF)")
    uploaded_pdf = st.file_uploader("Select a financial PDF", type=["pdf"])
    if uploaded_pdf is not None:
        if st.button("Ingest & Index PDF", use_container_width=True):
            with st.status(f"Parsing & indexing `{uploaded_pdf.name}`...", expanded=True) as status:
                try:
                    files = {
                        "file": (
                            uploaded_pdf.name,
                            uploaded_pdf.getvalue(),
                            "application/pdf",
                        )
                    }
                    res = requests.post(f"{API_BASE_URL}/ingest", files=files, timeout=300)
                    res.raise_for_status()
                    data = res.json()
                    status.update(
                        label=f"Indexed {data['chunks_indexed']} chunks from {data['source_file']}",
                        state="complete",
                    )
                    st.rerun()
                except Exception as exc:
                    status.update(label="Ingestion failed", state="error")
                    st.error(str(exc))

    st.divider()
    st.subheader("2. Retrieval Settings")
    indexed_files = fetch_documents()
    doc_options = ["All Indexed Filings"] + indexed_files
    selected_doc = st.selectbox("Scope search to filing:", options=doc_options)
    source_filter = None if selected_doc == "All Indexed Filings" else selected_doc

    top_n = st.slider("Top-N reranked chunks to LLM:", min_value=1, max_value=10, value=5)

    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# --- Main Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("guardrail_triggered"):
            st.warning(msg["content"])
        else:
            st.markdown(msg["content"])

        if msg.get("meta"):
            st.caption(msg["meta"])

        if msg.get("sources"):
            with st.expander(f"View {len(msg['sources'])} Retrieved Citations"):
                for idx, src in enumerate(msg["sources"], start=1):
                    st.markdown(
                        f"**[Source {idx}] `{src['source_file']}`** | "
                        f"type: `{src['chunk_type']}` | index: `{src['index']}` | "
                        f"rerank_score: `{src['rerank_score']}`"
                    )
                    st.caption(src["snippet"])

if prompt := st.chat_input("Ask a financial question (e.g., What was TCS FY26 revenue?)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving, reranking & synthesizing..."):
            try:
                payload = {
                    "query": prompt,
                    "source_file": source_filter,
                    "top_n": top_n,
                }
                resp = requests.post(f"{API_BASE_URL}/query", json=payload, timeout=120)

                if resp.status_code == 422:
                    st.session_state.messages.pop()
                    st.warning("Query must be between 3 and 1,000 characters.")
                else:
                    resp.raise_for_status()
                    result = resp.json()

                    answer = result["answer"]
                    guardrail = result["guardrail_triggered"]
                    sources = result["sources"]
                    meta = f"Model: `{result['model_used']}` | Scope: `{selected_doc}`"

                    if guardrail:
                        st.warning(answer)
                    else:
                        st.markdown(answer)
                    st.caption(meta)

                    if sources:
                        with st.expander(f"View {len(sources)} Retrieved Citations"):
                            for idx, src in enumerate(sources, start=1):
                                st.markdown(
                                    f"**[Source {idx}] `{src['source_file']}`** | "
                                    f"type: `{src['chunk_type']}` | index: `{src['index']}` | "
                                    f"rerank_score: `{src['rerank_score']}`"
                                )
                                st.caption(src["snippet"])

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "guardrail_triggered": guardrail,
                            "meta": meta,
                            "sources": sources,
                        }
                    )
            except Exception as exc:
                st.error(f"API Request Failed: {exc}")