# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

ner_tokenizer = AutoTokenizer.from_pretrained("Leo97/KoELECTRA-small-v3-modu-ner")
ner_model     = AutoModelForTokenClassification.from_pretrained("Leo97/KoELECTRA-small-v3-modu-ner")
ner_pipeline  = pipeline(
    "ner",
    model=ner_model,
    tokenizer=ner_tokenizer,
    aggregation_strategy="simple"
)
# special tokens 등록
ner_special_tokens = ["<TI>", "<PERSON>", "<LOCATION>", "<ORG>"]
ner_tokenizer.add_special_tokens({"additional_special_tokens": ner_special_tokens})
ner_model.resize_token_embeddings(len(ner_tokenizer))

NO_GENERALIZE_ENTITIES = ["CIVILIZATION"]

app = FastAPI()
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
emb_special_tokens = ner_special_tokens
# special tokens 등록(tokenizer)
model.tokenizer.add_special_tokens({"additional_special_tokens": emb_special_tokens})
model._first_module().auto_model.resize_token_embeddings(len(model.tokenizer))
dim = 768 # jhgan/ko-sroberta-multitask 모델의 임베딩 차원
index = faiss.IndexFlatIP(dim)

# 카테고리 목록 및 임베딩 전역 변수로 선언
CATEGORIES = [
    "가사", "취미", "휴식", "건강", "미용",
    "차량 관리", "반려 동물", "가족", "연애", "친목",
    "업무", "학업", "시험", "여행", "경제",
    "출장", "구매", "예약", "정기 지출", "재무",
    "세금", "봉사", "통화", "종교", "치료"
]
CATEGORY_EMBS = model.encode(CATEGORIES, convert_to_numpy=True, normalize_embeddings=True)

def ner_generalize_texts(texts: list[str], placeholder_map: dict[str, str] = None) -> list[str]:
    if placeholder_map is None:
        placeholder_map = {"PS": "<PERSON>", "LC": "<LOCATION>", "OG": "<ORG>"}
    generalized_list = []
    for text in texts:
        ents = ner_pipeline(text)
        new_text = text
        for ent in sorted(ents, key=lambda x: x["start"], reverse=True):
            # CV, TM, AF 타입인 경우 일반화 제외
            if ent["word"] in NO_GENERALIZE_ENTITIES or ent["entity_group"] == "CV" or ent["entity_group"] == "TM" or ent["entity_group"] == "AF":
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
    original_texts = s.texts
    generalized_texts = ner_generalize_texts(original_texts)
    print(f"Generalized: {generalized_texts}")

    embs = model.encode(generalized_texts, convert_to_numpy=True, normalize_embeddings=True)

    results = []
    for i, original_text in enumerate(original_texts):
        text_emb = embs[i]
        # 전역 변수로 선언된 CATEGORY_EMBS 사용
        similarities_to_categories = np.dot(text_emb, CATEGORY_EMBS.T)

        selected_categories = []
        # 유사도가 0.5 이상인 카테고리 선택
        for cat_idx, sim_score in enumerate(similarities_to_categories):
            if sim_score >= 0.5:
                selected_categories.append(CATEGORIES[cat_idx]) # 전역 변수 CATEGORIES 사용

        # 0.5 이상인 카테고리가 없는 경우
        if not selected_categories:
            max_similarity_score = np.max(similarities_to_categories) # 가장 높은 유사도 점수 확인
            closest_category_index = np.argmax(similarities_to_categories)
            selected_categories.append(CATEGORIES[closest_category_index]) # 전역 변수 CATEGORIES 사용

        results.append({
            "original_text": original_text,
            "embedding": text_emb.tolist(),
            "categories": selected_categories
        })

    return {"results": results}

print(embed(Sentences(texts=[ "엔진오일 교환", "민수랑 8시 술약속", "오전 10시 주간회의", "지윤이와 롯데몰 8시 데이트 약속", "가족들과 저녁식사", "민수랑 차량 정비"])))