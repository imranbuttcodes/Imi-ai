# Nexus AI — The Multi-Agent Framework

Nexus AI is a plug-and-play, modular AI architecture built with **LangChain**, **LangGraph**, and **Pydantic**. It serves as an expandable framework where specialist AI agents can be dynamically loaded and routed to without hardcoding.

## Core Architecture (The Router Pattern)

The system is built on the **Main AI Router Pattern**.
The philosophy of Nexus AI is strict separation of concerns:
- **Infrastructure never knows about business logic.**
- **Supervisor (Main AI) never writes answers if a specialist is better.**
- **Agents never decide which agent to call next.**

### The Execution Flow
1. **User Input:** The user types a message in the terminal (`app.py`).
2. **Main AI (The Front Door):** The `main_ai_node` receives the prompt. It reads the `registry` and dynamically generates a prompt listing all available specialist agents. It outputs a structured JSON decision: answer directly, or route to a specific agent.
3. **The Router (Train Switch):** The `route_request` function acts as a conditional edge in LangGraph. It shifts the graph execution to the chosen agent (or ends it).
4. **Specialist Agent:** The chosen agent (e.g., Knowledge Agent) receives the query, runs its internal logic (like RAG), updates the global `NexusState`, and finishes.

---

## Project Structure

```text
Nexus-AI/
├── .env                  # API keys and strategy toggles (groq, cohere, crag/self_rag)
├── app.py                # The Terminal CLI & Entry Point
├── requirements.txt      # Project dependencies
├── data/
│   ├── documents/        # Drop raw PDFs and text files here
│   ├── chroma/           # The local vector database (auto-generated)
│   └── ingest.py         # Script to chunk, hash, and embed documents into ChromaDB
├── core/
│   ├── config.py         # Pydantic BaseSettings (reads .env automatically)
│   ├── constants.py      # System-wide static variables
│   ├── state.py          # NexusState (TypedDict) — the shared memory object for the graph
│   ├── registry.py       # Auto-discovery system for agents
│   ├── main_ai.py        # The routing LLM node
│   └── router.py         # LangGraph conditional edge logic
├── infrastructure/
│   ├── llm/
│   │   └── factory.py    # Provider-agnostic LLM instantiator (get_llm)
│   └── retrieval/
│       ├── embeddings.py # Cohere/OpenAI/HF embedding factory
│       ├── vector_store.py# BM25 + ChromaDB Ensemble Retriever
│       ├── crag.py       # Corrective RAG pipeline (Grade -> Rewrite -> Web Search)
│       └── self_rag.py   # Self-Reflective RAG pipeline (Hallucination & Usefulness checks)
├── graphs/
│   └── nexus_graph.py    # The Master Blueprint (dynamically wires Main AI to agents)
└── agents/
    ├── base.py           # The ABC contract all agents must inherit from
    ├── knowledge/        # [Specialist] The Knowledge Agent
    │   ├── agent.py      # Registers the agent and exposes its run() method
    │   └── nodes.py      # The internal logic (routes to Basic, CRAG, or Self-RAG)
    └── research/         # [Specialist] The Deep Web Research Agent
        ├── agent.py
        ├── graph.py
        └── nodes.py
```

---

## The Agent Ecosystem

Adding a new agent to Nexus AI requires **zero** changes to the core routing logic. 
If you create `agents/coder/agent.py` and inherit from `BaseAgent`, the system will automatically discover it on boot, add it to the Main AI's prompt, and wire it into the LangGraph circuit board.

### 1. The Knowledge Agent
Acts as an internal expert on private company documents (PDFs/txt).
**Flow:**
- Reads `settings.rag_strategy` from `.env`.
- Dynamically executes either:
  - **Basic RAG:** Direct retrieval and generation.
  - **CRAG (Corrective RAG):** Evaluates retrieved documents. If they are bad, it uses Tavily to search the live web. It filters the text sentence-by-sentence before generating an answer.
  - **Self-RAG:** A highly complex, looping LangGraph that grades documents, checks its own generated answers for hallucinations against the context, and checks if the answer actually resolved the user's question. It rewrites and retries if it fails.

### 2. The Research Agent
Acts as a deep-web researcher capable of breaking down complex questions and searching the live internet.
**Flow:**
- **Plan:** Uses an LLM to break the complex user query down into 2-3 specific Google search queries.
- **Search:** Executes the queries using the Tavily API and accumulates the raw text results.
- **Report:** Synthesizes the raw data into a highly structured markdown report with strict source citations.

---

## Core Infrastructure Deep Dive

- **`core.config.Settings`**: Reads `.env` and provides strongly-typed configuration. No hardcoded API keys anywhere.
- **`core.registry.AgentRegistry`**: A singleton that scans the `agents/` folder. It uses `importlib` to dynamically load any class inheriting from `BaseAgent`.
- **`infrastructure.llm.factory.get_llm()`**: Agents never import `ChatGroq` or `ChatOpenAI`. They ask the factory for an LLM. The factory uses LangChain's `init_chat_model` and explicitly passes the `model_provider` to prevent schema errors, keeping the code 100% provider-agnostic.
- **`graphs.nexus_graph`**: Uses a dynamic loop. It adds the `main_ai_node`, loops through every agent in the registry, adds their `.run()` methods as nodes, and creates a conditional routing map.

---

## How to Run

1. **Setup Environment:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
2. **Configure `.env`:**
   Copy `.env.example` to `.env` and insert your API keys (Groq, Cohere, Tavily).
3. **Ingest Documents:**
   Place PDFs in `data/documents/` and run:
   ```powershell
   python data/ingest.py
   ```
4. **Start the System:**
   ```powershell
   python app.py
   ```
