# Personal Knowledge Assistant

A complete AI project that covers **LangChain, RAG, AI Agents, MCP, and LLMOps**. RAG mode runs 100% locally for free using Ollama. Agent mode requires an Anthropic API key (paid, per use).

---

## What You'll Learn

| Feature | File | What it teaches |
|---------|------|-----------------|
| **LangChain** | `src/app.py` | Chains, prompts, memory, retrievers |
| **RAG** | `src/ingest.py` + `app.py` | Document loading, chunking, embeddings, vector search |
| **AI Agents** | `src/tools.py` + `app.py` | Tool use, ReAct reasoning, decision-making |
| **MCP** | `src/mcp_server.py` | Protocol design, tool schemas, client-server pattern |
| **LLMOps** | `src/llmops.py` | Logging, retries, grounding evaluation |

---

## Setup (Step by Step)

### Step 1: Install Ollama
Download and install from: https://ollama.com

### Step 2: Start Ollama and pull models
```bash
# Start the Ollama server (keep this running)
ollama serve

# In a new terminal, pull the models we need
ollama pull llama3.2           # The main LLM (~2GB)
ollama pull nomic-embed-text   # For creating embeddings (~270MB)
```

### Step 3: Set up the Python environment
```bash
# Navigate to this project folder
cd knowledge-assistant

# Create a virtual environment
python -m venv venv

# Activate it
# On Mac/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 4: Set up your API key (for Agent Mode only)
```bash
# Copy the example file
cp .env.example .env

# Open .env and replace the placeholder with your real key
# Get your key at: https://console.anthropic.com
ANTHROPIC_API_KEY=your-anthropic-api-key-here
```
> Skip this step if you only plan to use RAG Mode (fully local, no API key needed).

### Step 5: Ingest your documents into the vector store
```bash
python src/ingest.py
```
This reads all `.txt` files in the `docs/` folder, splits them into chunks, creates embeddings, and saves everything to ChromaDB. **Run this once** (or whenever you add new documents).

### Step 6: Launch the app!
```bash
streamlit run src/app.py
```
Open your browser to: http://localhost:8501

---

## How to Use

### 📚 RAG Mode (Ask your docs)
Ask questions about the documents in your `docs/` folder:
- "What is RAG and how does it work?"
- "What are the different types of machine learning?"
- "How do I use pip to install packages?"
- "What is ChromaDB used for?"

The AI will search your documents and answer based only on what's in them.

### 🤖 Agent Mode (Use tools)
The AI will decide which tool to use:
- "What time is it right now?" → uses `get_current_datetime` tool
- "What is 2847 * 193?" → uses `calculate` tool
- "Count the words in: The quick brown fox" → uses `get_word_count` tool
- "What is RAG?" → uses `search_documents` tool

### 🔌 MCP Explorer
Explore the MCP server interactively — see available tools, their schemas, and make live tool calls.

---

## Project Structure

```
knowledge-assistant/
├── docs/                    # Your documents go here
│   ├── ai_basics.txt        # Sample: AI concepts
│   ├── python_basics.txt    # Sample: Python fundamentals
│   └── langchain_rag.txt    # Sample: LangChain & RAG
│
├── src/
│   ├── app.py               #  Main Streamlit app (LangChain + RAG + Agent)
│   ├── ingest.py            #  Document ingestion pipeline (RAG setup)
│   ├── tools.py             #  Agent tools (Agents)
│   ├── mcp_server.py        #  MCP server implementation
│   └── llmops.py            #  Logging, retries, evaluation (LLMOps)
│
├── chroma_db/               # Created after running ingest.py
├── llmops_log.jsonl         # Created after first query
└── requirements.txt
```

---

## Try Adding Your Own Documents

1. Add any `.txt` file to the `docs/` folder
2. Run `python src/ingest.py` again to re-index
3. Ask questions about your new content!

---

## Troubleshooting

**"Connection refused" error**
→ Make sure Ollama is running: `ollama serve`

**"Model not found" error**
→ Pull the models: `ollama pull llama3.2` and `ollama pull nomic-embed-text`

**"Collection not found" error**
→ Run ingestion first: `python src/ingest.py`

**Slow responses**
→ Normal for local LLMs. llama3.2 is fast; larger models are slower but smarter.

**Want a smarter model?**
→ Try `ollama pull mistral` or `ollama pull llama3.1:8b`, then update `RAG_MODEL` in `app.py`

---

## How Each Concept Connects

```
User Question
     │
     ▼
[LangChain] orchestrates everything
     │
     ├─► [RAG] Search ChromaDB → retrieve relevant chunks
     │         └─► Inject into prompt → LLM answers from YOUR data
     │
     ├─► [Agent] LLM reasons: "which tool do I need?"
     │         └─► Calls tools (calculator, clock, doc search)
     │
     ├─► [MCP] Standard protocol: tools have schemas, servers expose them
     │         └─► Any MCP client can use any MCP server
     │
     └─► [LLMOps] Log the call, check grounding, retry on failure
               └─► Dashboard shows latency, quality over time
```
