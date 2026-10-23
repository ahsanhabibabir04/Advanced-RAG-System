
"""Streamlit interface for the Advanced RAG System."""

import streamlit as st

from generation import generate_answer
from ingestion import chunk_pdf_pages
from retrieval import HybridRetriever


st.set_page_config(
    page_title="Advanced RAG System",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Advanced RAG System")
st.caption(
    "Ask questions about your PDFs using hybrid search and a local LLM."
)

with st.sidebar:
    st.header("Settings")
    top_k = st.slider(
        "Retrieved passages",
        min_value=1,
        max_value=10,
        value=5,
    )
    st.caption("LLM: qwen2.5:3b")
    st.caption("Retrieval: FAISS + BM25 + RRF")

uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
)

if "documents" not in st.session_state:
    st.session_state.documents = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.button("Process PDFs", type="primary"):
    if not uploaded_files:
        st.warning("Please upload at least one PDF.")
    else:
        all_chunks = []
        failed_files = []

        with st.spinner("Extracting PDF text..."):
            for uploaded_file in uploaded_files:
                try:
                    # Save the uploaded PDF to a temporary local file.
                    import tempfile
                    from pathlib import Path

                    with tempfile.TemporaryDirectory() as temp_dir:
                        pdf_path = Path(temp_dir) / uploaded_file.name
                        pdf_path.write_bytes(uploaded_file.getvalue())
                        file_chunks = chunk_pdf_pages(pdf_path)

                    all_chunks.extend(file_chunks)

                except Exception as exc:
                    failed_files.append(
                        f"{uploaded_file.name}: {exc}"
                    )

        if failed_files:
            for error in failed_files:
                st.error(error)

        if not all_chunks:
            st.error(
                "No readable text was found. The PDF may be scanned "
                "or contain no extractable text."
            )
        else:
            with st.spinner("Building hybrid search index..."):
                retriever = HybridRetriever()
                retriever.fit(
                    [chunk["text"] for chunk in all_chunks]
                )

            st.session_state.documents = all_chunks
            st.session_state.retriever = retriever
            st.session_state.chat_history = []

            st.success(
                f"Processed {len(uploaded_files)} PDF file(s) "
                f"into {len(all_chunks)} text chunks."
            )

if st.session_state.retriever is not None:
    st.divider()
    st.subheader("Ask your documents")

    question = st.text_input(
        "Your question",
        placeholder="What are the main findings in these documents?",
    )

    if st.button("Get Answer"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching documents..."):
                results = st.session_state.retriever.search(
                    question,
                    top_k=top_k,
                )

            # Restore source metadata for the retrieved text chunks.
            source_chunks = []
            for result in results:
                source_chunks.append(
                    st.session_state.documents[result["chunk_index"]]
                )

            with st.spinner("Generating answer with Ollama..."):
                try:
                    answer = generate_answer(question, source_chunks)
                    st.session_state.chat_history.append(
                        {
                            "question": question,
                            "answer": answer,
                            "sources": source_chunks,
                        }
                    )
                except Exception as exc:
                    st.error(
                        "Could not generate an answer. Confirm that "
                        "Ollama is running and qwen2.5:3b is available."
                    )
                    st.exception(exc)

    for item in reversed(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(item["question"])

        with st.chat_message("assistant"):
            st.markdown(item["answer"])

            with st.expander("Retrieved sources"):
                for source in item["sources"]:
                    st.markdown(
                        f"**{source['source']} — Page {source['page']}**"
                    )
                    st.write(source["text"])
                    st.divider()
else:
    st.info("Upload one or more PDFs, then click **Process PDFs**.")
