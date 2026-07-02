import os
import re
import uuid
import pickle
import textwrap
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

import fitz  # PyMuPDF
import numpy as np
import faiss
from groq import Groq
from langdetect import detect
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langgraph.graph import StateGraph, END

# --- Load Environment Variables ---
def load_env_vars():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split("=", 1)
                if len(parts) == 2:
                    key, val = parts
                    os.environ[key.strip()] = val.strip()

load_env_vars()

# --- Initialize FastAPI App ---
app = FastAPI(title="Research Paper Assistant Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global In-Memory Stores ---
ACTIVE_PAPERS: Dict[str, Dict[str, Any]] = {}
LANGUAGE_NAMES = {
    "en": "English", "ml": "Malayalam", "hi": "Hindi",
    "ta": "Tamil", "te": "Telugu", "kn": "Kannada",
    "fr": "French", "de": "German", "es": "Spanish",
    "zh-cn": "Chinese", "ar": "Arabic", "ja": "Japanese",
}

# --- PDF Processing & Vector Database ---
def extract_pdf_text(path: Path) -> List[Dict[str, Any]]:
    """Extract text from every page of the PDF using PyMuPDF."""
    doc = fitz.open(str(path))
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    doc.close()
    return pages

def chunk_pages(pages: List[Dict[str, Any]], size: int = 350, step: int = 150) -> tuple[List[str], List[Dict[str, Any]]]:
    """Sliding-window word chunker. Returns (chunks, metadata)."""
    chunks, meta = [], []
    for page_info in pages:
        words = page_info["text"].split()
        page_num = page_info["page"]
        for i in range(0, len(words), step):
            chunk = " ".join(words[i : i + size])
            if len(chunk.strip()) < 60:
                continue
            chunks.append(chunk)
            meta.append({"page": page_num, "offset": i})
    return chunks, meta

class FAISSStore:
    """TF-IDF + LSA -> L2-normalised dense vectors -> FAISS IndexFlatIP (cosine similarity)."""
    def __init__(self, chunks: List[str], meta: List[Dict[str, Any]], dim: int = 256):
        self.chunks = chunks
        self.meta = meta
        self.vectorizer = TfidfVectorizer(
            max_features=80000,
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        X_sp = self.vectorizer.fit_transform(chunks)
        real_dim = min(dim, X_sp.shape[1] - 1, X_sp.shape[0] - 1)
        self.dim = max(1, real_dim)  # Make sure dim is at least 1
        
        self.svd = TruncatedSVD(n_components=self.dim, random_state=42)
        X_dense = self.svd.fit_transform(X_sp).astype(np.float32)
        faiss.normalize_L2(X_dense)
        
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(X_dense)

    def _embed(self, text: str) -> np.ndarray:
        x = self.svd.transform(self.vectorizer.transform([text])).astype(np.float32)
        faiss.normalize_L2(x)
        return x

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        scores, idxs = self.index.search(self._embed(query), k)
        return [
            {
                "text": self.chunks[i],
                "score": float(scores[0][r]),
                "page": self.meta[i]["page"],
            }
            for r, i in enumerate(idxs[0]) if i != -1
        ]

# --- Groq LLM Wrapper ---
class GroqLLM:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Groq API key must be provided.")
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"

    def chat(self, system: str, user: str, max_tokens: int = 700) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content.strip()

# --- LangGraph QA Pipeline ---
class ChatState(BaseModel):
    query: str = Field(...)
    paper_id: str = Field(...)
    api_key: str = Field(...)
    lang_code: str = "en"
    lang_name: str = "English"
    action: Optional[str] = None
    docs: Optional[List[Dict[str, Any]]] = None
    answer: Optional[str] = None

PLANNER_SYS = textwrap.dedent("""
    You are a routing assistant for a Research Paper QA system.
    The user may write in any language.
    Decide whether the question needs searching the uploaded research paper
    for specific details, definitions, data, authors, or arguments (action = "retrieve")
    or can be answered from general knowledge / greetings / off-topic chats (action = "answer_direct").

    Rules:
    - Questions about the paper's contents, methodology, findings, terms, figures, authors, or datasets → retrieve
    - Greetings, general knowledge questions, completely off-topic banter → answer_direct

    Reply with ONLY one word: retrieve   OR   answer_direct
""").strip()

ANSWER_SYS = textwrap.dedent("""
    You are a helpful research assistant. You answer questions about the uploaded
    research paper STRICTLY based on the provided context excerpts.

    CRITICAL RULE: You MUST reply in {lang_name} ({lang_code}).
    Match the user's language exactly. If the query is in Spanish, reply in Spanish; if in Malayalam, reply in Malayalam, etc.

    Guidelines:
    - Be concise, clear, and academic.
    - Quote page numbers when referencing facts (e.g., "on page 4, the authors...").
    - If the context does not contain enough information to answer, state this clearly in {lang_name}.
    - Do NOT make up findings, statistics, or details.
""").strip()

FALLBACK_SYS = textwrap.dedent("""
    You are a helpful research assistant.
    Reply in {lang_name} ({lang_code}).
    If the question is off-topic, politely redirect the user back to discussing the research paper.
""").strip()

def node_detect_language(state: ChatState) -> ChatState:
    try:
        code = detect(state.query)
    except Exception:
        code = "en"
    name = LANGUAGE_NAMES.get(code, code.upper())
    state.lang_code = code
    state.lang_name = name
    return state

def node_planner(state: ChatState) -> ChatState:
    llm = GroqLLM(api_key=state.api_key)
    routing_prompt = (
        f"User question (may be in {state.lang_name}): {state.query}\n\n"
        "Route this: reply ONLY with 'retrieve' or 'answer_direct'."
    )
    decision = llm.chat(PLANNER_SYS, routing_prompt, max_tokens=8)
    state.action = "retrieve" if "retrieve" in decision.lower() else "answer_direct"
    return state

def node_retrieve(state: ChatState) -> ChatState:
    paper = ACTIVE_PAPERS.get(state.paper_id)
    if paper:
        vector_store = paper["vector_store"]
        # Search for query
        results = vector_store.search(state.query, k=5)
        state.docs = results
    else:
        state.docs = []
    return state

def node_answer(state: ChatState) -> ChatState:
    llm = GroqLLM(api_key=state.api_key)
    sys_prompt = ANSWER_SYS.format(
        lang_name=state.lang_name, lang_code=state.lang_code
    ) if state.docs else FALLBACK_SYS.format(
        lang_name=state.lang_name, lang_code=state.lang_code
    )

    if state.docs:
        passages = "\n\n".join(
            f"[Page {d['page']} | relevance {d['score']:.2f}]\n{d['text']}"
            for d in state.docs
        )
        user_msg = (
            f"Context from the uploaded research paper:\n{passages}\n\n"
            f"Question ({state.lang_name}): {state.query}"
        )
    else:
        user_msg = state.query

    state.answer = llm.chat(sys_prompt, user_msg, max_tokens=700)
    return state

# --- Compile StateGraph ---
builder = StateGraph(ChatState)
builder.add_node("detect_language", node_detect_language)
builder.add_node("planner",         node_planner)
builder.add_node("retriever",       node_retrieve)
builder.add_node("answer",          node_answer)

builder.set_entry_point("detect_language")
builder.add_edge("detect_language", "planner")
builder.add_conditional_edges(
    "planner",
    lambda s: s.action,
    {"retrieve": "retriever", "answer_direct": "answer"},
)
builder.add_edge("retriever", "answer")
builder.add_edge("answer",    END)

graph_app = builder.compile()

# --- Helpers to Get API Key ---
def get_groq_api_key(header_key: Optional[str] = None) -> str:
    key = header_key or os.environ.get("GROQ_API_KEY")
    if not key or key == "YOUR_GROQ_API_KEY_HERE":
        raise HTTPException(
            status_code=400,
            detail="Groq API key not set. Set it in .env or supply it in the request."
        )
    return key

# --- API Endpoints ---
@app.post("/upload")
async def upload_paper(
    file: UploadFile = File(...),
    api_key: Optional[str] = Form(None)
):
    try:
        # Resolve API key
        groq_key = get_groq_api_key(api_key)
        
        # Save temp file
        paper_id = str(uuid.uuid4())
        upload_dir = Path("temp_uploads")
        upload_dir.mkdir(exist_ok=True)
        temp_path = upload_dir / f"{paper_id}.pdf"
        
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # Extract text & chunk
        pages = extract_pdf_text(temp_path)
        if not pages:
            raise HTTPException(status_code=400, detail="The PDF contains no readable text.")
            
        chunks, meta = chunk_pages(pages)
        if not chunks:
            raise HTTPException(status_code=400, detail="The PDF is too small or couldn't be chunked.")

        # Build vector store
        vector_store = FAISSStore(chunks, meta)
        
        # Store in active papers
        ACTIVE_PAPERS[paper_id] = {
            "title": file.filename,
            "page_count": len(pages),
            "vector_store": vector_store,
            "path": temp_path
        }

        # Generate Initial Analysis
        llm = GroqLLM(api_key=groq_key)

        # 1. Summary
        sum_docs = vector_store.search("abstract introduction overview summary conclusion", k=6)
        sum_context = "\n\n".join(d["text"] for d in sum_docs)
        summary_prompt = f"Based on the following excerpts, write a clear, structured summary of this paper including its main goals, methodology, and key findings:\n\n{sum_context}"
        summary = llm.chat("You are a helpful research assistant writing a paper summary.", summary_prompt, max_tokens=600)

        # 2. Contributions
        contrib_docs = vector_store.search("contributions results findings novelty value", k=5)
        contrib_context = "\n\n".join(d["text"] for d in contrib_docs)
        contrib_prompt = f"Based on the following excerpts, list the main scientific and technical contributions of this paper. Present them as bullet points:\n\n{contrib_context}"
        contributions = llm.chat("You are a helpful research assistant listing paper contributions.", contrib_prompt, max_tokens=400)

        # 3. Technical Terms
        terms_docs = vector_store.search("methodology technical equations definition framework acronyms", k=5)
        terms_context = "\n\n".join(d["text"] for d in terms_docs)
        terms_prompt = f"Based on the following excerpts, identify 5-8 advanced technical terms, algorithms, frameworks, or acronyms used in the paper and explain them briefly in bullet points:\n\n{terms_context}"
        technical_terms = llm.chat("You are a helpful research assistant defining key terms.", terms_prompt, max_tokens=500)

        return {
            "paper_id": paper_id,
            "title": file.filename,
            "page_count": len(pages),
            "summary": summary,
            "contributions": contributions,
            "technical_terms": technical_terms
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

class QueryRequest(BaseModel):
    query: str
    paper_id: str
    api_key: Optional[str] = None

@app.post("/query")
async def query_paper(req: QueryRequest):
    try:
        groq_key = get_groq_api_key(req.api_key)
        
        if req.paper_id not in ACTIVE_PAPERS:
            raise HTTPException(status_code=404, detail="Paper not found. Please upload it first.")

        # Run state graph QA pipeline
        inputs = {
            "query": req.query,
            "paper_id": req.paper_id,
            "api_key": groq_key
        }
        
        result = graph_app.invoke(inputs)
        return {
            "answer": result.get("answer"),
            "lang_code": result.get("lang_code"),
            "lang_name": result.get("lang_name"),
            "action": result.get("action"),
            "pages": list(set(d["page"] for d in (result.get("docs") or [])))
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok", "active_papers": len(ACTIVE_PAPERS)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
