import os
import tempfile
import uuid

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "qwen/qwen3.8-27b"

SYSTEM_PROMPT = """
You are a helpful assistant that answers questions about the user's uploaded documents.
 
Follow these rules:
1. Facts specific to the documents (values, units, reference ranges, names, dates, figures,
   statuses) must come ONLY from the context below. Never invent or guess them. If the context
   does not contain them, say so clearly.
2. If the user asks a general or conceptual question (for example "what is X?", "what does Y
   mean?", "why does Z matter?"), explain it using your general knowledge, even if the context
   does not define it. Start with a short, clear explanation. Say that this part is general
   background rather than taken from the documents.
3. When the context also has related information (such as the user's own value for that item),
   add it after the explanation and mention which file it came from.
4. For health or other professional topics, keep explanations factual and suggest consulting a
   qualified professional for interpretation of personal results.
5. Be concise.
 
context: {context}
"""

st.set_page_config(page_title="Document Q&A (RAG)", page_icon="📚", layout="wide")
st.title("📚 Chat with your documents")
st.caption("Upload PDF, TXT and CSV files, process them, then ask questions.")


# ---------- cached resources ----------
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


@st.cache_resource
def get_llm():
    return ChatGroq(model=LLM_MODEL, temperature=0, reasoning_format="parsed")


# ---------- helpers ----------
def load_uploaded_file(uploaded_file):
    """Save an uploaded file to a temp path, load it, and tag docs with the real filename."""
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    try:
        if ext == ".pdf":
            docs = PyPDFLoader(tmp_path).load()
        elif ext == ".csv":
            docs = CSVLoader(tmp_path, encoding="utf-8").load()
        elif ext == ".txt":
            docs = TextLoader(tmp_path, encoding="utf-8", autodetect_encoding=True).load()
        else:
            return []
    finally:
        os.remove(tmp_path)

    for d in docs:
        d.metadata["source"] = uploaded_file.name  # replace temp path with original name
    return docs


def build_vector_store(files, chunk_size, chunk_overlap):
    all_docs, per_file = [], {}
    for f in files:
        try:
            docs = load_uploaded_file(f)
        except Exception as e:
            st.warning(f"Could not read **{f.name}**: {e}")
            continue
        docs = [d for d in docs if d.page_content.strip()]
        if not docs:
            st.warning(f"**{f.name}** has no extractable text (scanned PDF or empty file?).")
            continue
        per_file[f.name] = len(docs)
        all_docs.extend(docs)

    if not all_docs:
        return None, {}, 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(all_docs)

    store = Chroma.from_documents(
        chunks,
        get_embeddings(),
        collection_name=f"docs_{uuid.uuid4().hex[:8]}",  # fresh collection on every rebuild
    )
    return store, per_file, len(chunks)


def format_docs(docs):
    return "\n\n---\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}"
        + (f", page {d.metadata['page'] + 1}" if "page" in d.metadata else "")
        + f"]\n{d.page_content}"
        for d in docs
    )


def answer_question(question, k):
    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(question)
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{question}")])
    chain = prompt | get_llm() | StrOutputParser()
    answer = chain.invoke({"context": format_docs(docs), "question": question})
    return answer, docs


# ---------- session state ----------
st.session_state.setdefault("vector_store", None)
st.session_state.setdefault("messages", [])
st.session_state.setdefault("index_info", None)

# ---------- sidebar ----------
with st.sidebar:
    st.header("1. Upload documents")
    files = st.file_uploader(
        "PDF, TXT or CSV files",
        type=["pdf", "txt", "csv"],
        accept_multiple_files=True,
    )

    st.header("2. Settings")
    chunk_size = st.slider("Chunk size (characters)", 200, 2000, 500, step=50)
    chunk_overlap = st.slider("Chunk overlap", 0, 500, 100, step=10)
    top_k = st.slider("Chunks to retrieve (k)", 1, 10, 3)

    process = st.button("Process documents", type="primary", disabled=not files)

    if process:
        if chunk_overlap >= chunk_size:
            st.error("Overlap must be smaller than chunk size.")
        else:
            with st.spinner("Reading, chunking and embedding..."):
                store, per_file, n_chunks = build_vector_store(files, chunk_size, chunk_overlap)
            if store is None:
                st.error("No usable text found in the uploaded files.")
            else:
                st.session_state.vector_store = store
                st.session_state.index_info = {"files": per_file, "chunks": n_chunks}
                st.session_state.messages = []  # new knowledge base, fresh chat
                st.success("Ready! Ask your questions.")

    info = st.session_state.index_info
    if info:
        st.divider()
        st.subheader("Indexed")
        for name, n in info["files"].items():
            st.write(f"• {name} ({n} page/row docs)")
        st.write(f"**{info['chunks']} chunks** in the vector store")

    if st.session_state.messages and st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------- chat ----------
if st.session_state.vector_store is None:
    st.info("👈 Upload one or more files and click **Process documents** to get started.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for s in msg["sources"]:
                        st.markdown(f"**{s['label']}**")
                        st.text(s["text"])

    question = st.chat_input("Ask a question about your documents")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching and thinking..."):
                try:
                    answer, docs = answer_question(question, top_k)
                except Exception as e:
                    answer, docs = f"Something went wrong: {e}", []
            st.markdown(answer)

            sources = []
            for i, d in enumerate(docs, 1):
                label = f"{i}. {d.metadata.get('source', 'unknown')}"
                if "page" in d.metadata:
                    label += f" (page {d.metadata['page'] + 1})"
                sources.append({"label": label, "text": d.page_content})
            if sources:
                with st.expander("Sources"):
                    for s in sources:
                        st.markdown(f"**{s['label']}**")
                        st.text(s["text"])

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )