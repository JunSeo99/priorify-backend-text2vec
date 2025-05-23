from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

app = FastAPI()
model = SentenceTransformer('jhgan/ko-sroberta-multitask')

# NER 구성
ner_tokenizer = AutoTokenizer.from_pretrained("Leo97/KoELECTRA-small-v3-modu-ner")
ner_model     = AutoModelForTokenClassification.from_pretrained("Leo97/KoELECTRA-small-v3-modu-ner")
ner_pipeline  = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy="simple")

ner_special_tokens = ["<TI>", "<PERSON>", "<LOCATION>", "<ORG>"]
ner_tokenizer.add_special_tokens({"additional_special_tokens": ner_special_tokens})
ner_model.resize_token_embeddings(len(ner_tokenizer))
model.tokenizer.add_special_tokens({"additional_special_tokens": ner_special_tokens})
model._first_module().auto_model.resize_token_embeddings(len(model.tokenizer))

NO_GENERALIZE_ENTITIES = ["CIVILIZATION"]

def ner_generalize_texts(texts: list[str], placeholder_map: dict[str, str] = None) -> list[str]:
    if placeholder_map is None:
        placeholder_map = {"PS": "<PERSON>", "LC": "<LOCATION>", "OG": "<ORG>"}
    generalized_list = []
    for text in texts:
        ents = ner_pipeline(text)
        new_text = text
        for ent in sorted(ents, key=lambda x: x["start"], reverse=True):
            if ent["word"] in NO_GENERALIZE_ENTITIES or ent["entity_group"] in ["CV", "TM", "AF"]:
                continue
            label = ent["entity_group"]
            token = placeholder_map.get(label, f"<{label}>")
            start, end = ent["start"], ent["end"]
            new_text = new_text[:start] + token + new_text[end:]
        generalized_list.append(new_text)
    return generalized_list

class Sentences(BaseModel):
    texts: list[str]

@app.post("/embed")
def embed(s: Sentences):
    generalized_texts = ner_generalize_texts(s.texts)
    embs = model.encode(generalized_texts, convert_to_numpy=True, normalize_embeddings=True)
    return {"embeddings": embs.tolist()}