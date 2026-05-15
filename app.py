"""
Legal RAG Contract Risk Analyzer — FastAPI Backend
Analyzes legal contracts using Retrieval-Augmented Generation (RAG)
with FLAN-T5 for LLM reasoning and FAISS for vector search.
"""

import os

# Prevent MacOS multiprocessing segfaults with HuggingFace models
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import json
import uuid
import logging
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import PyPDF2
import pdfplumber

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
VECTOR_DIR = BASE_DIR / "vector_store"
STATIC_DIR = BASE_DIR / "static"
MODEL_DIR = BASE_DIR  # FLAN-T5 model lives in project root

UPLOAD_DIR.mkdir(exist_ok=True)
VECTOR_DIR.mkdir(exist_ok=True)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 5
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("legal-rag")

# ---------------------------------------------------------------------------
# App initialization
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Legal RAG Contract Risk Analyzer",
    description="AI-powered legal contract analysis using RAG",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global model holders — loaded synchronously at startup
# ---------------------------------------------------------------------------
embedding_model = None
t5_tokenizer = None
t5_model = None

# In-memory document registry:  doc_id -> metadata dict
documents: dict = {}

@app.on_event("startup")
def load_models():
    global embedding_model, t5_tokenizer, t5_model
    from sentence_transformers import SentenceTransformer
    from transformers import T5ForConditionalGeneration, T5Tokenizer

    logger.info("Loading Sentence Transformer embedding model …")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    logger.info("Embedding model loaded.")

    logger.info("Loading FLAN-T5 model from %s …", MODEL_DIR)
    t5_tokenizer = T5Tokenizer.from_pretrained(str(MODEL_DIR))
    t5_model = T5ForConditionalGeneration.from_pretrained(str(MODEL_DIR))
    t5_model.eval()
    logger.info("FLAN-T5 model loaded.")
    
    _restore_documents()

def ensure_models_loaded():
    pass

def _restore_documents():
    """Re-load document metadata from vector_store on restart."""
    for meta_path in VECTOR_DIR.glob("*_meta.json"):
        doc_id = meta_path.stem.replace("_meta", "")
        with open(meta_path) as f:
            documents[doc_id] = json.load(f)
    logger.info("Restored %d document(s) from disk.", len(documents))


# ---------------------------------------------------------------------------
# Helpers — PDF text extraction
# ---------------------------------------------------------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text with pdfplumber; fall back to PyPDF2."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception:
        logger.warning("pdfplumber failed, falling back to PyPDF2")

    if not text.strip():
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

    return text.strip()


# ---------------------------------------------------------------------------
# Helpers — Chunking (manual implementation to avoid langchain dep issues)
# ---------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks using paragraph/sentence boundaries."""
    separators = ["\n\n", "\n", ". ", " "]
    chunks = []

    def _split_recursive(txt: str, sep_idx: int = 0) -> list[str]:
        if len(txt) <= chunk_size:
            return [txt] if txt.strip() else []

        if sep_idx >= len(separators):
            # Hard split as last resort
            result = []
            for i in range(0, len(txt), chunk_size - overlap):
                piece = txt[i:i + chunk_size]
                if piece.strip():
                    result.append(piece.strip())
                if i + chunk_size >= len(txt):
                    break
            return result

        sep = separators[sep_idx]
        parts = txt.split(sep)
        result = []
        current = ""

        for part in parts:
            candidate = (current + sep + part) if current else part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current.strip():
                    result.append(current.strip())
                if len(part) > chunk_size:
                    result.extend(_split_recursive(part, sep_idx + 1))
                    current = ""
                else:
                    current = part
        if current.strip():
            result.append(current.strip())
        return result

    raw_chunks = _split_recursive(text)

    # Add overlap between chunks
    final_chunks = []
    for i, chunk in enumerate(raw_chunks):
        if i > 0 and overlap > 0:
            prev_tail = raw_chunks[i - 1][-overlap:]
            chunk = prev_tail + " " + chunk
        final_chunks.append(chunk.strip())

    return final_chunks if final_chunks else [text[:chunk_size]]


# ---------------------------------------------------------------------------
# Helpers — Embedding & FAISS
# ---------------------------------------------------------------------------
def build_faiss_index(chunks: list[str], doc_id: str):
    """Embed chunks and persist a FAISS index + metadata."""
    ensure_models_loaded()
    embeddings = embedding_model.encode(chunks, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    index_path = str(VECTOR_DIR / f"{doc_id}.index")
    faiss.write_index(index, index_path)

    meta = {
        "chunks": chunks,
        "num_chunks": len(chunks),
        "embedding_dim": dim,
    }
    with open(VECTOR_DIR / f"{doc_id}_meta.json", "w") as f:
        json.dump(meta, f)

    return meta


def search_faiss(query: str, doc_id: str, top_k: int = TOP_K):
    """Retrieve top-k relevant chunks for a query."""
    ensure_models_loaded()
    index_path = str(VECTOR_DIR / f"{doc_id}.index")
    if not os.path.exists(index_path):
        raise FileNotFoundError("FAISS index not found for this document.")

    index = faiss.read_index(index_path)

    with open(VECTOR_DIR / f"{doc_id}_meta.json") as f:
        meta = json.load(f)

    query_embedding = embedding_model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx < len(meta["chunks"]):
            results.append({
                "rank": rank + 1,
                "chunk_index": int(idx),
                "distance": float(dist),
                "text": meta["chunks"][idx],
            })
    return results


# ---------------------------------------------------------------------------
# Helpers — LLM analysis
# ---------------------------------------------------------------------------
def analyze_with_llm(query: str, context_chunks: list[dict]) -> str:
    """Use FLAN-T5 to analyze retrieved contract clauses."""
    ensure_models_loaded()
    context_text = "\n\n".join(
        f"[Clause {c['rank']}]: {c['text']}" for c in context_chunks
    )

    prompt = (
        "You are a legal contract risk analyst. Analyze the following contract clauses "
        "retrieved from a legal document and answer the user's question.\n\n"
        "For each relevant clause, identify:\n"
        "1. Risk Level (High / Medium / Low)\n"
        "2. Key concerns\n"
        "3. Recommendations\n\n"
        f"Contract Clauses:\n{context_text}\n\n"
        f"Question: {query}\n\n"
        "Provide a detailed risk analysis with specific references to the clauses above."
    )

    inputs = t5_tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    )

    with torch.no_grad():
        outputs = t5_model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=1.5,
        )

    response = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------
@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF contract, extract text, chunk, embed, and index."""
    if not file.filename.lower().endswith((".pdf", ".PDF")):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    doc_id = str(uuid.uuid4())[:8]
    save_path = UPLOAD_DIR / f"{doc_id}_{file.filename}"

    # Save uploaded file
    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Extract text
    logger.info("Extracting text from %s …", file.filename)
    text = extract_text_from_pdf(str(save_path))
    if not text:
        os.remove(save_path)
        raise HTTPException(status_code=400, detail="Could not extract text from the PDF.")

    # Chunk
    chunks = chunk_text(text)
    logger.info("Created %d chunks from %s", len(chunks), file.filename)

    # Build FAISS index
    meta = build_faiss_index(chunks, doc_id)

    # Register document
    doc_info = {
        "doc_id": doc_id,
        "filename": file.filename,
        "file_path": str(save_path),
        "num_chunks": meta["num_chunks"],
        "text_length": len(text),
        "chunks": chunks,
    }
    documents[doc_id] = doc_info

    return {
        "status": "success",
        "doc_id": doc_id,
        "filename": file.filename,
        "num_chunks": meta["num_chunks"],
        "text_length": len(text),
        "message": f"Document processed successfully — {meta['num_chunks']} chunks indexed.",
    }


@app.post("/api/query")
async def query_document(
    doc_id: str = Form(...),
    query: str = Form(...),
):
    """Query a specific document for risk analysis."""
    if doc_id not in documents:
        raise HTTPException(status_code=404, detail="Document not found.")

    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # Retrieve relevant chunks
    retrieved = search_faiss(query, doc_id, top_k=TOP_K)

    if not retrieved:
        raise HTTPException(status_code=404, detail="No relevant clauses found.")

    # LLM analysis
    analysis = analyze_with_llm(query, retrieved)

    return {
        "status": "success",
        "doc_id": doc_id,
        "query": query,
        "analysis": analysis,
        "retrieved_clauses": retrieved,
        "num_clauses": len(retrieved),
    }


@app.get("/api/documents")
async def list_documents():
    """List all uploaded documents."""
    docs = []
    for doc_id, info in documents.items():
        docs.append({
            "doc_id": doc_id,
            "filename": info["filename"],
            "num_chunks": info["num_chunks"],
            "text_length": info["text_length"],
        })
    return {"documents": docs}


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document and its index."""
    if doc_id not in documents:
        raise HTTPException(status_code=404, detail="Document not found.")

    info = documents.pop(doc_id)

    # Clean up files
    try:
        os.remove(info["file_path"])
    except FileNotFoundError:
        pass
    try:
        os.remove(str(VECTOR_DIR / f"{doc_id}.index"))
    except FileNotFoundError:
        pass
    try:
        os.remove(str(VECTOR_DIR / f"{doc_id}_meta.json"))
    except FileNotFoundError:
        pass

    return {"status": "success", "message": f"Document {doc_id} deleted."}


# ---------------------------------------------------------------------------
# Serve frontend
# ---------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def serve_frontend():
    return FileResponse(str(STATIC_DIR / "index.html"))


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
