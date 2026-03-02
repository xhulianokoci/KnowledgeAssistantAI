"""
PERSONAL KNOWLEDGE ASSISTANT
================================
  RAG    -> Ollama + ChromaDB (local)
  Agent  -> Claude API
  MCP    -> protocol explorer
  LLMOps -> logging, latency, grounding
"""

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from dotenv import load_dotenv
load_dotenv()

import time
import streamlit as st
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langgraph.prebuilt import create_react_agent
from langchain_core.tools.retriever import create_retriever_tool
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache

set_llm_cache(InMemoryCache())

from tools import ALL_TOOLS
from mcp_server import mcp_server
from llmops import log_llm_call, get_session_stats, get_recent_logs

# ── Configuration ──────────────────────────────────────────────────────────────
RAG_MODEL   = "llama3.2"
AGENT_MODEL = "claude-haiku-4-5-20251001"
EMBED_MODEL = "nomic-embed-text"
CHROMA_DIR  = "chroma_db"
# ───────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Knowledge Assistant", page_icon="🧠", layout="wide")
st.title("Personal Knowledge Assistant")
st.caption("Ollama · ChromaDB (local) · Claude API · MCP · LLMOps")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    mode = st.radio(
        "Query Mode",
        ["📚 RAG Mode", "🕵🏻 Agent Mode", "🔌 MCP Explorer"],
        help="RAG uses Ollama locally. Agent uses Claude API."
    )

    if "🕵🏻 Agent" in mode:
        st.divider()
        if os.getenv("ANTHROPIC_API_KEY"):
            st.success("🔑 API key loaded from .env")
        else:
            st.error("🔑 ANTHROPIC_API_KEY not found in .env")

    st.divider()
    st.subheader("📊 LLMOps Dashboard")
    if st.button("Refresh Stats"):
        st.session_state["stats_refresh"] = True

    stats = get_session_stats()
    if "message" in stats:
        st.info(stats["message"])
    else:
        c1, c2 = st.columns(2)
        c1.metric("Queries", stats["total_queries"])
        c2.metric("Avg Latency", f"{stats['avg_latency_ms']}ms")
        st.metric("Avg Grounding", f"{stats['avg_grounding_score']:.0%}")

    st.divider()
    st.subheader("📋 Recent Logs")
    for log in reversed(get_recent_logs(3)):
        with st.expander(f"🔍 {log['query'][:40]}..."):
            st.json({
                "latency": f"{log['latency_ms']}ms",
                "grounding": f"{log['grounding_score']:.0%}",
                "chunks": log['chunks_retrieved'],
                "mode": log['mode']
            })

# ── Model Loaders ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_rag_llm():
    return ChatOllama(model=RAG_MODEL, temperature=0.1)


@st.cache_resource
def load_vectorstore():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)


@st.cache_resource
def load_rag_chain(_llm, _vectorstore):
    retriever = _vectorstore.as_retriever(search_kwargs={"k": 3})

    rag_prompt = ChatPromptTemplate.from_messages([
        ("human", (
            "Answer the question based only on the context below.\n"
            "If the answer is not in the context, say you do not have that information.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}"
        ))
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | _llm
        | StrOutputParser()
    )
    return chain, retriever


@st.cache_resource
def load_agent(api_key, _vectorstore):
    from langchain_anthropic import ChatAnthropic

    llm = ChatAnthropic(model=AGENT_MODEL, api_key=api_key, temperature=0)

    retriever = _vectorstore.as_retriever(search_kwargs={"k": 2})
    doc_tool = create_retriever_tool(
        retriever,
        name="search_documents",
        description="Search knowledge base docs about AI, Python, ML, business, or finance."
    )

    all_tools = ALL_TOOLS + [doc_tool]

    system_prompt = (
        "You are a document assistant. Use tools to help the user.\n\n"
        "Tool input formats:\n"
        "- list_documents: pass empty string\n"
        "- read_document: pass filename e.g. ai_basics.txt\n"
        "- search_in_document: pass filename.txt | search term\n"
        "- append_to_document: pass filename.txt | text to add\n"
        "- replace_text_in_document: pass filename.txt | old text | new text\n"
        "- get_current_datetime: pass empty string\n"
        "- search_documents: pass your search query"
    )

    return create_react_agent(llm, all_tools, prompt=system_prompt)


# ── Chat History ───────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📎 Sources used"):
                for i, doc in enumerate(message["sources"], 1):
                    src = doc.metadata.get("source", "Unknown")
                    st.markdown(f"**Source {i}:** `{src}`")
                    st.markdown(f"> {doc.page_content[:300]}...")

# ── MCP Explorer ───────────────────────────────────────────────────────────────
if "🔌 MCP Explorer" in mode:
    st.subheader("🔌 MCP Server Explorer")
    st.info("MCP (Model Context Protocol) — standard way for AI models to connect to tools.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Server Info**")
        st.json(mcp_server.get_info())
        st.markdown("**Available Tools**")
        for tool_def in mcp_server.list_tools():
            with st.expander(f"🔧 `{tool_def['name']}`"):
                st.write(tool_def["description"])
                st.code(str(tool_def["inputSchema"]), language="json")

    with col2:
        st.markdown("**Try a Tool Call**")
        selected_tool = st.selectbox("Select tool", ["list_documents", "read_document"])
        if selected_tool == "read_document":
            filename = st.text_input("Filename", "ai_basics.txt")
            args = {"filename": filename}
        else:
            args = {}
        if st.button("📡 Call Tool"):
            result = mcp_server.call_tool(selected_tool, args)
            if "error" in result:
                st.error(result["error"])
            else:
                content = result.get("content", "")
                display = content[:500] + ("..." if len(content) > 500 else "")
                st.text_area("Result", display, height=200)
    st.stop()

# ── Chat Input ─────────────────────────────────────────────────────────────────
query = st.chat_input("Ask a question...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                vectorstore = load_vectorstore()
                start_time = time.time()

                if "📚 RAG" in mode:
                    rag_llm = load_rag_llm()
                    rag_chain, retriever = load_rag_chain(rag_llm, vectorstore)
                    source_docs = retriever.invoke(query)
                    answer = rag_chain.invoke(query)
                    latency = (time.time() - start_time) * 1000

                    log_entry = log_llm_call(
                        query=query,
                        retrieved_chunks=source_docs,
                        answer=answer,
                        model=RAG_MODEL,
                        latency_ms=latency,
                        mode="rag"
                    )

                    st.markdown(answer)
                    if source_docs:
                        with st.expander("📎 Sources used"):
                            for i, doc in enumerate(source_docs, 1):
                                src = doc.metadata.get("source", "Unknown")
                                st.markdown(f"**Source {i}:** `{src}`")
                                st.markdown(f"> {doc.page_content[:300]}...")

                    c1, c2 = st.columns(2)
                    c1.caption(f"⚡ {latency:.0f}ms · 🏠 Ollama")
                    c2.caption(f"📊 Grounding: {log_entry['grounding_score']:.0%}")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": source_docs
                    })

                else:
                    api_key = os.getenv("ANTHROPIC_API_KEY", "")
                    if not api_key:
                        st.error("ANTHROPIC_API_KEY not found. Add it to your .env file and restart.")
                        st.stop()

                    agent_executor = load_agent(api_key, vectorstore)
                    result = agent_executor.invoke({"messages": [("human", query)]})
                    answer = result["messages"][-1].content
                    latency = (time.time() - start_time) * 1000

                    log_llm_call(
                        query=query,
                        retrieved_chunks=[],
                        answer=answer,
                        model=AGENT_MODEL,
                        latency_ms=latency,
                        mode="agent"
                    )

                    st.markdown(answer)
                    st.caption(f"⚡ {latency:.0f}ms · ☁️ Claude API")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

            except Exception as e:
                error_msg = str(e)
                if "connect" in error_msg.lower() or "connection refused" in error_msg.lower():
                    st.error("Cannot connect to Ollama. Run `ollama serve` first.")
                elif "chroma" in error_msg.lower() or "no such file" in error_msg.lower():
                    st.error("Vector store not found. Run `python src/ingest.py` first.")
                elif "authentication" in error_msg.lower() or "401" in error_msg:
                    st.error("Invalid API key. Check console.anthropic.com")
                else:
                    st.error(f"Error: {error_msg}")