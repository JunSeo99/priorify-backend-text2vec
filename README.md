# Priorify - AI Backend (Text2Vec & NER)

> “누구나 J가 될 수 있도록”  
> 일정 관리의 복잡함을 그래프로 시각화하여, 어떤 일을 먼저 처리해야 할지 명확하게 알려주는 AI 기반 스케줄링 서비스

[![Backend](https://img.shields.io/badge/service-Text2Vec-green)]()



---

[![🖥️ Frontend](https://img.shields.io/badge/Frontend-Next.js-151515?style=for-the-badge&logo=next.js&logoColor=white)](https://github.com/JunSeo99/priorify-backend-frontend)

[![🖥️ Backend API](https://img.shields.io/badge/API-SpringBoot-6DB33F?style=for-the-badge&logo=springboot&logoColor=white)]([<YOUR_BACKEND_URL>](https://github.com/JunSeo99/priorify-backend-backend))

[![⚡ FastAPI Server](https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://github.com/JunSeo99/priorify-backend-text2vec)


## 🚀 프로젝트 소개 (Project Overview)

Priorify는 사용자의 스케줄 텍스트를 분석하여 연관성을 파악하고, 동적으로 우선순위를 결정해주는 지능형 스케줄 관리 서비스입니다. 전체 서비스는 다음과 같은 4단계 핵심 로직으로 동작합니다.

1. **텍스트 임베딩** (Text2Vec + NER)  
2. **유사도 기반 Vector Search** (MongoDB Atlas Vector Search)  
3. **그래프 시각화 & 카테고리 자동 분류**  
4. **우선순위 자동 결정**  

해당 레포지토리는 FastAPI와 Text2Vec 모델을 사용하여 텍스트 임베딩 기능을 제공하는 AI 백엔드입니다.

---

## 📦 프론트엔드 기술 스택

- **프레임워크**: FastAPI  
- **언어**: Python 3.10+  
- **모델**:  
  - Text2Vec (jhgan/ko-sroberta-multitask)  
  - KoELECTRA NER (Leo97/KoELECTRA-small-v3-modu-ner)  
- **서빙**: Uvicorn

### 🚀 필수 기능

* **일정 텍스트를 받아 Text2Vec 모델로 임베딩**
* * **개체명 인식(NER) 전처리**
* **KoELECTRA 기반 NER로 고유명사 태깅**

---

### 📆 개발 일정

| Week | Milestone                              |
| :--: | :------------------------------------- |
|   1  | Project selection & initial research   |
|   2  | Repo fork & environment setup          |
|   3  | Git branching strategy & CI pipeline   |
|   4  | Authentication & basic to-do endpoints |
|   5  | AI-driven auto-categorization          |
|   6  | Integration with D3.js visualization   |
|   7  | UI/UX polish & end-to-end testing      |
|   8  | Final review & documentation           |

---

### 👥 팀원

* **PM / Front-end**: 송준서 (32202337, Department of Software, Dankook University)
* **Back-end**: 윤치호 (32227546, Department of Software, Dankook University)
* **Back-end**: 이지훈 (32243528, Department of Software, Dankook University)

---

### 🚦 Git Workflow

1. Fork the repository
2. Create feature branch: `feature/<name>`
3. Open Pull Request
4. Code review & merge into develop
5. Merge develop into main
