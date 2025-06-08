# app.py
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import os # os.path.exists를 위해 추가

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
model = SentenceTransformer('jhgan/ko-sroberta-multitask')

emb_special_tokens = ner_special_tokens
# special tokens 등록(tokenizer)
model.tokenizer.add_special_tokens({"additional_special_tokens": emb_special_tokens})
model._first_module().auto_model.resize_token_embeddings(len(model.tokenizer), mean_resizing=False)
dim = 768 # jhgan/ko-sroberta-multitask dimention

CATEGORIES_DEFINITIONS = {
    "가사": ["집안일", "청소", "빨래", "요리", "장보기", "정리정돈"], # 4시까지 집안일 하기: [0.91, 0] <=유사도 비교=> a + b + c + d / 4
    "휴식": ["휴식 취하기", "잠자기", "음악감상", "영화보기", "드라마 시청", "게임하기", "멍때리기"],
    "건강": ["운동", "병원", "건강검진", "약", "영양", "담배", ],
    "미용": ["미용실 가기", "피부과 방문", "화장품 구매", "네일아트", "헤어 관리"],
    "차량 관리": ["주유하기", "세차하기", "자동차 정비", "차량 검사", "타이어 교체"],
    "반려 동물": ["반려동물 산책시키기", "사료 구매", "간식 주기", "동물병원 가기", "목욕시키기"],
    "가족": ["부모님", "가족", "아이", "아들", "딸", "아버지", "어머니", "동생", "친척"],
    "연애": ["데이트", "애인", "기념일"],
    "친목": ["친구", "모임", "동호회", "술자리"],
    "업무": ["회사일", "미팅", "회의", "보고서 작성", "이메일 확인", "업무 통화", "야근"],
    "학업": ["공부하기", "수업 듣기", "강의 수강", "과제 제출", "리포트 작성", "시험 준비", "논문 쓰기", "스터디 모임", "자격증 공부", "온라인 강의"],
    "시험": ["시험 응시", "자격증 시험", "운전면허 시험", "토익 시험"],
    "여행": ["국내 여행", "해외 여행", "여행 계획 세우기", "숙소 예약", "항공권 예매"],
    "출장": ["업무 출장", "지방 출장", "해외 출장"],
    "구매": ["온라인 쇼핑", "오프라인 쇼핑", "물건 사기", "장바구니 결제"],
    "예약": ["식당 예약", "병원 예약", "미용실 예약", "티켓 예매"],
    "정기 지출": ["월세 납부", "공과금 납부", "관리비 납부", "대출이자 상환", "보험료 납부"],
    "재무": ["계좌 이체", "은행 업무", "가계부 작성", "주식 투자", "펀드 관리"],
    "세금": ["세금 납부", "연말정산", "종합소득세 신고"],
    "보험": ["보험금 청구", "보험 계약"],
    "봉사": ["봉사활동 참여", "기부하기"],
    "통화": ["전화 통화", "영상 통화"],
    "종교": ["종교 활동", "예배 참석", "미사 참석", "성경 공부"],
    "치료": ["물리치료 받기", "상담 치료", "정신과 진료"],
}


CATEGORY_EMBEDDING_FILE = "category_avg_embeddings.npy"

# 기존 CATEGORIES 및 CATEGORY_EMBS 정의는 새로운 방식으로 대체됨
# CATEGORIES = [ ... ]
# CATEGORY_EMBS = model.encode(CATEGORIES, convert_to_numpy=True, normalize_embeddings=True)
# print("카테고리 임베딩 완료")


def save_category_embeddings(
        categories_definitions: dict[str, list[str]],
        model_instance: SentenceTransformer,
        output_path: str
):
    """
    카테고리 정의(키워드 리스트)를 바탕으로 각 카테고리의 평균 임베딩을 계산하고 파일에 저장합니다.
    """
    category_embeddings_map = {}
    for category, keywords in categories_definitions.items():
        if not keywords:
            print(f"Warning: Category '{category}' has no keywords. Skipping.")
            continue

        keyword_embs = model_instance.encode(keywords, convert_to_numpy=True, normalize_embeddings=True)
        avg_emb = np.mean(keyword_embs, axis=0)
        # 평균 임베딩도 정규화 (일관성을 위해)
        if np.linalg.norm(avg_emb) > 0: # 0으로 나누는 것 방지
            avg_emb = avg_emb / np.linalg.norm(avg_emb)

        category_embeddings_map[category] = avg_emb
        print(f"카테고리 '{category}' 평균 임베딩 계산 완료.")

    np.save(output_path, category_embeddings_map)
    print(f"카테고리 평균 임베딩이 '{output_path}'에 저장되었습니다.")

def calculate_similarity_with_saved_categories(
        generalized_text: str,
        model_instance: SentenceTransformer,
        category_embedding_path: str
) -> tuple[dict[str, float], list[str], list[float]]:
    """
    입력된 일반화 텍스트를 임베딩하고, 저장된 카테고리 평균 임베딩과 유사도를 계산합니다.
    유사도 딕셔너리, 카테고리 이름 리스트, 유사도 점수 리스트를 반환합니다.
    """
    if not os.path.exists(category_embedding_path):
        raise FileNotFoundError(f"카테고리 임베딩 파일 '{category_embedding_path}'을 찾을 수 없습니다. 먼저 save_category_embeddings 함수를 실행하거나 서버를 재시작하여 자동 생성되도록 하세요.")

    category_embeddings_map = np.load(category_embedding_path, allow_pickle=True).item()

    if not category_embeddings_map:
        raise ValueError(f"'{category_embedding_path}' 파일에서 카테고리 임베딩을 로드할 수 없거나 비어있습니다.")

    text_emb = model_instance.encode(generalized_text, convert_to_numpy=True, normalize_embeddings=True)

    similarities = {}
    loaded_categories = list(category_embeddings_map.keys())
    loaded_category_embs_list = [category_embeddings_map[cat] for cat in loaded_categories]

    if not loaded_category_embs_list:
        raise ValueError("로드된 카테고리 임베딩이 비어있습니다.")

    loaded_category_embs_np = np.array(loaded_category_embs_list)

    similarity_scores = np.dot(loaded_category_embs_np, text_emb).tolist()

    for i, category_name in enumerate(loaded_categories):
        similarities[category_name] = similarity_scores[i]

    return similarities, loaded_categories, similarity_scores

def ner_generalize_texts(texts: list[str], placeholder_map: dict[str, str] = None) -> list[str]:
    if placeholder_map is None:
        placeholder_map = {"PS": "<PERSON>", "LC": "<LOCATION>", "OG": "<ORG>"}
    generalized_list = []
    for text in texts:
        ents = ner_pipeline(text)
        new_text = text
        for ent in sorted(ents, key=lambda x: x["start"], reverse=True):
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

    results = []
    for i, original_text in enumerate(original_texts):
        current_generalized_text = generalized_texts[i]

        try:
            all_similarities_map, loaded_categories_list, similarities_to_categories_scores = calculate_similarity_with_saved_categories(
                current_generalized_text,
                model,
                CATEGORY_EMBEDDING_FILE
            )
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return {"error": str(e), "message": "카테고리 임베딩 파일이 준비되지 않았습니다. 서버 로그를 확인하거나 관리자에게 문의하세요."}
        except ValueError as e:
            print(f"Error: {e}")
            return {"error": str(e), "message": "카테고리 임베딩 데이터에 문제가 있습니다."}

        # 응답에 포함될 텍스트의 임베딩 (정규화 안 함)
        text_emb_for_response = model.encode(current_generalized_text, convert_to_numpy=True, normalize_embeddings=False)

        selected_categories = []
        similarity_threshold = 0.5 # 임계값, 필요시 조정

        if similarities_to_categories_scores: # 유사도 점수가 있을 경우에만 로직 수행
            for cat_idx, sim_score in enumerate(similarities_to_categories_scores):
                if sim_score >= similarity_threshold:
                    selected_categories.append(loaded_categories_list[cat_idx])

            if not selected_categories: # 임계값 이상인 카테고리가 없는 경우
                max_similarity_score = np.max(similarities_to_categories_scores)
                closest_category_index = np.argmax(similarities_to_categories_scores)
                selected_categories.append(loaded_categories_list[closest_category_index])
        else: # 유사도 점수 자체가 없는 극단적인 경우 (예: 로드된 카테고리가 없을 때)
            selected_categories.append("분류 불가")

        results.append({
            "original_text": original_text,
            "generalized_text": current_generalized_text,
            "embedding": text_emb_for_response.tolist(),
            "categories": selected_categories,
            # "all_similarities": all_similarities_map / 모든 카테고리와의 유사도 포함
        })

    return {"results": results}


if __name__ == "__main__":
    # 서버 시작 시 카테고리 임베딩 파일이 없으면 생성
    if not os.path.exists(CATEGORY_EMBEDDING_FILE):
        print(f"'{CATEGORY_EMBEDDING_FILE}'을 찾을 수 없어 새로 생성합니다.")
        save_category_embeddings(CATEGORIES_DEFINITIONS, model, CATEGORY_EMBEDDING_FILE)
    else:
        print(f"'{CATEGORY_EMBEDDING_FILE}'이 이미 존재합니다. 기존 파일을 사용합니다.")
    uvicorn.run(app, host="0.0.0.0", port=8000)

