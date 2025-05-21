# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

app = FastAPI()
model = SentenceTransformer('all-MiniLM-L6-v2')
dim = 384
index = faiss.IndexFlatIP(dim)

class Sentences(BaseModel):
    texts: list[str]

@app.post("/embed")
def embed(s: Sentences):
    embs = model.encode(s.texts, convert_to_numpy=True, normalize_embeddings=True)
    return {"embeddings": embs.tolist()}