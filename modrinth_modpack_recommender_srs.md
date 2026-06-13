# SRS: Modrinth 모드팩 추천 시스템

## 1. 목적

Modrinth에서 수집한 모드팩 CSV 데이터를 기반으로, 사용자가 자연어로 원하는 조건을 입력하면 관련 모드팩을 추천하는 프로그램을 만든다.

추천은 다음 요소를 사용한다.

```text
전처리된 description 텍스트
tags/categories 메타 토큰
Word2Vec 기반 검색어 확장
TF-IDF 벡터화
코사인 유사도
다운로드 수 기반 인기도 보정
메타데이터 필터
```

## 2. 입력 데이터

입력 파일:

```text
./datasets/modrinth_dataset.csv
```

입력 CSV는 `modrinth_dataset.py`가 생성한 파일을 기본으로 한다.

필수 컬럼:

```text
name
slug
url
description
body
tags
categories
loaders
game_versions
client_side
server_side
license
downloads
followers
date_created
date_modified
```

`description`은 짧은 설명이고, `body`는 긴 상세 설명이다.

## 3. 출력 데이터

### 3.1 전처리 CSV

출력 파일:

```text
./datasets/modrinth_dataset_preprocessed.csv
```

요구사항:

```text
원본 CSV의 모든 기존 컬럼을 유지한다.
원본 컬럼 순서를 그대로 유지한다.
새 컬럼이 필요하면 원본 컬럼 뒤에만 추가한다.
description 컬럼은 기존 위치를 유지하되 값만 추천용 전처리 텍스트로 교체한다.
body, tags, categories, downloads, loaders 등 필터용 컬럼은 원본 그대로 유지한다.
```

### 3.2 모델/행렬 출력

`generate_model.py`는 다음 파일을 만든다.

```text
./models/modpack_word2vec.model
./models/tfidf_vectorizer.pkl
./models/tfidf_matrix.npz
./models/modpack_meta.csv
```

`modpack_meta.csv` 필수 컬럼:

```text
index
name
slug
url
tags
categories
loaders
game_versions
downloads
followers
date_created
date_modified
```

## 4. 전처리 요구사항

### 4.1 원문 설명 구성

각 row의 번역/전처리 원문은 다음과 같이 만든다.

```text
source_text = description + "\n\n" + body
```

둘 중 하나가 비어 있으면 존재하는 텍스트만 사용한다.

출력 CSV에서는 `body`를 변경하지 않고, `description`만 전처리 결과로 교체한다.

### 4.2 HTML/Markdown 정리

번역 전에 `source_text`에서 HTML/Markdown/링크 노이즈를 정리한다.

요구사항:

```text
HTML block 태그는 줄바꿈 또는 문단 경계로 변환한다.
HTML inline 태그는 텍스트만 보존한다.
script/style/iframe/svg/video/audio/picture/img/pre/code/kbd/samp는 제거한다.
Markdown 이미지는 제거한다.
Markdown 링크는 URL을 제거하고 링크 텍스트만 보존한다.
raw URL은 제거한다.
코드 블록은 제거한다.
표 구분선과 장식용 구분선은 제거한다.
목록 항목은 줄 단위 텍스트로 보존한다.
```

### 4.3 언어 감지

`source_text`의 언어를 감지한다.

우선순위:

```text
1. 한글/가나/CJK 문자권 힌트로 ko/ja/zh 보정
2. langdetect detect_langs 확률로 주 언어 판단
3. 실패 시 unknown
```

반환 언어 예시:

```text
en
ko
ja
zh
unknown
```

### 4.4 번역 규칙

최종 추천 인덱스는 영어 단일 텍스트 공간으로 만든다.

전처리 단계에서는 원문이 영어가 아닌 경우에만 영어로 번역한다. 원문이 이미 영어이면 번역하지 않는다.

중요 요구사항:

```text
절대 2단 번역 금지.
문서 전체를 영어/한국어 양방향으로 번역하지 않는다.
모든 문서 번역은 원문 source_text → English 직접 번역만 허용한다.
검색용 최종 description은 영어 전처리 토큰 문자열 하나만 가진다.
```

언어별 처리:

```text
원문이 영어(en):
    english_text = source_text

원문이 한국어(ko):
    english_text = source_text → English 직접 번역

원문이 일본어/중국어(ja/zh):
    english_text = source_text → English 직접 번역

원문이 unknown:
    english_text = source_text
    로그를 남기고 번역은 시도하지 않는다.
```

TranslateGemma는 Hugging Face `transformers`의 모델 카드/공식 문서 방식으로 호출한다.

참조 공식 문서:

```text
https://huggingface.co/google/translategemma-4b-it
https://huggingface.co/docs/transformers/main_classes/text_generation
https://huggingface.co/docs/transformers/quantization/bitsandbytes
https://huggingface.co/docs/huggingface_hub/guides/download
```

기본 모델 ID:

```python
TRANSLATE_MODEL_ID = "google/translategemma-4b-it"
```

모델 파일 저장 위치:

```text
./models/<safe_model_id>
```

예:

```text
./models/google__translategemma-4b-it
```

양자화 설정:

```python
TRANSLATE_QUANTIZATION = "8bit"  # none, 8bit, 4bit
MODEL_DTYPE = "float16"
BNB_4BIT_COMPUTE_DTYPE = "bfloat16"
```

양자화는 별도 양자화 모델을 다운로드하는 방식이 아니다. 이미 받은 safetensors를 `BitsAndBytesConfig`로 로드 시점에 양자화한다.

TranslateGemma chat template 입력:

```python
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "source_lang_code": source_lang_code,
                "target_lang_code": target_lang_code,
                "text": text_to_translate,
            }
        ],
    }
]
```

`source_lang_code`와 `target_lang_code`는 `en`, `ko`, `ja`, `zh` 같은 언어 코드로 넘긴다.

### 4.5 번역 토큰 예산

TranslateGemma 입력 context가 작으므로, 문자 수가 아니라 tokenizer/processor 기준 토큰 수로 제한한다.

기본값:

```python
MAX_TRANSLATE_INPUT_TOKENS = 1024
HARD_MAX_TRANSLATE_INPUT_TOKENS = 1800
MAX_TRANSLATE_OUTPUT_TOKENS = 2000
```

`MAX_TRANSLATE_INPUT_TOKENS`는 번역 모델에 한 번에 넣는 원문 chunk의 최대 입력 토큰 수다.

`HARD_MAX_TRANSLATE_INPUT_TOKENS`는 단일 문장이 일반 chunk 예산을 넘을 때도 번역을 시도할 수 있는 절대 입력 상한이다. 단일 문장이 이 값까지 넘으면 모델 입력 한계를 넘는 것으로 보고 해당 문장을 건너뛴다.

`MAX_TRANSLATE_OUTPUT_TOKENS`는 모델이 새로 생성할 수 있는 번역문 토큰 수의 상한이다. 이 값은 출력문을 자르기 위한 상한이지, 반드시 그만큼 생성한다는 뜻은 아니다. 단, 입력 chunk에 비해 너무 작으면 번역문이 중간에 끊길 수 있으므로 `MAX_TRANSLATE_INPUT_TOKENS`와 균형 있게 잡는다.

긴 설명 처리:

```text
1. source_text를 빈 줄 기준 문단 단위로 분리한다.
2. 문단을 순서대로 묶되, chat template 적용 후 입력 토큰 수가 MAX_TRANSLATE_INPUT_TOKENS 이하가 되게 한다.
3. 단일 문단이 MAX_TRANSLATE_INPUT_TOKENS를 초과하면 문장 단위로 나눈다.
4. 단일 문장이 MAX_TRANSLATE_INPUT_TOKENS를 초과해도 HARD_MAX_TRANSLATE_INPUT_TOKENS 이하이면 단일 chunk로 번역을 시도한다.
5. 단일 문장이 HARD_MAX_TRANSLATE_INPUT_TOKENS도 초과하면 해당 문장은 건너뛰고 로그를 남긴다.
6. chunk 번역 결과는 원래 순서를 유지해 다시 합친다.
```

앞부분만 잘라 번역하는 방식은 금지한다.

### 4.6 번역 캐시

번역 결과는 캐시한다.

```text
./datasets/translation_cache.json
```

캐시 키에는 다음 값을 포함한다.

```text
source text hash
source language
target language
TRANSLATE_MODEL_ID
TRANSLATE_QUANTIZATION
MAX_TRANSLATE_INPUT_TOKENS
MAX_TRANSLATE_OUTPUT_TOKENS
```

문서 번역 캐시는 source_text → English 결과를 저장한다.

전처리 중단 후 재개는 최종 출력 CSV의 `complete` 컬럼으로 판단한다.

```text
./datasets/modrinth_dataset_preprocessed.csv
```

출력 CSV에는 원본 컬럼 뒤에 다음 컬럼을 추가한다.

```text
complete
```

`complete=1`이면 전처리 완료 row로 본다. `complete=0`, 빈 값, 컬럼 없음은 미완료로 보고 재실행 시 다시 전처리한다.

전처리기는 row 처리가 끝날 때마다 `modrinth_dataset_preprocessed.csv`에 append한다. 정상 완료 row는 `complete=1`, 번역/전처리 실패 row는 `complete=0`으로 기록한다.

전처리 종료 시 이번 실행에서 새로 성공한 row 수와 실패한 row 수를 출력한다.

### 4.7 텍스트 전처리

최종 `description`에 들어갈 텍스트는 영어 기준 전처리를 거친 공백 기준 토큰 문자열로 만든다.

기본 규칙:

```text
소문자화
URL 제거
마크다운 잔여 문법 제거
영문, 숫자, +, #, _, -, . 정도만 보존
영어 형태소/토큰 분리
불용어 제거
검색 의미가 약한 일반어 제거
LemmInflect 기반 원형 변환(lemmatization)
모드명/로더명/버전/메타 토큰 보존
```

문서와 검색어는 모두 같은 영어 전처리 규칙을 사용한다. 한국어 형태소 분석은 최종 인덱스가 영어 단일 공간이므로 기본 요구사항에서 제외한다.

### 4.8 tags/categories 메타 토큰

`tags`, `categories` 컬럼은 쉼표 기준으로 파싱한다.

예:

```text
tags = "Kitchen Sink, Optimization"
categories = "kitchen-sink, optimization"
```

메타 토큰 형식:

```text
tag:optimization
category:fabric
```

정규화:

```text
소문자화
공백은 _로 변환
중복 토큰 제거
```

삽입 규칙:

```text
영어 토큰열의 시작에 meta_tokens 삽입
영어 토큰열의 시작으로부터 20, 40, 60...번째 토큰 뒤에 meta_tokens 삽입
영어 토큰열의 끝에 meta_tokens 삽입
meta token은 불용어 제거와 원형 변환 대상에서 제외
```

## 5. Word2Vec 요구사항

### 5.1 목적

Word2Vec은 최종 추천 벡터를 직접 만드는 용도가 아니다.

역할:

```text
사용자 검색어 토큰을 의미적으로 확장하는 보조 모델
```

예:

```text
automation → tech, factory, create, mekanism
quest → questing, progression, guide
```

### 5.2 학습 데이터

학습 입력:

```text
전처리 CSV의 description 컬럼
```

각 description을 공백 기준으로 split하여 토큰 리스트로 사용한다.

### 5.3 학습 파라미터

기본값:

```python
Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=10,
    min_count=2,
    workers=4,
    sg=1,
    epochs=20
)
```

주의:

```text
name/title은 Word2Vec 학습에 넣지 않는다.
희귀 고유명사가 너무 많아지면 most_similar 품질이 깨질 수 있다.
```

### 5.4 저장

```text
./models/modpack_word2vec.model
```

## 6. TF-IDF 요구사항

### 6.1 목적

TF-IDF는 모든 모드팩 설명을 숫자 벡터 행렬로 변환한다.

추천 시 사용자의 검색어도 같은 vectorizer로 transform하여 코사인 유사도를 계산한다.

### 6.2 학습 데이터

```text
전처리 CSV의 description 컬럼
```

### 6.3 Vectorizer 설정

권장 기본값:

```python
TfidfVectorizer(
    tokenizer=str.split,
    preprocessor=None,
    token_pattern=None,
    lowercase=False,
    min_df=2,
    max_df=0.85,
    sublinear_tf=True,
    norm="l2"
)
```

요구사항:

```text
이미 전처리된 공백 분리 토큰 문자열이므로 tokenizer=str.split 사용
lowercase=False
token_pattern=None
fit_transform은 학습 시에만 사용
검색어에는 transform만 사용
```

### 6.4 저장

```text
./models/tfidf_vectorizer.pkl
./models/tfidf_matrix.npz
```

## 7. 추천 요구사항

### 7.1 검색 입력

사용자는 자연어 검색어를 입력한다.

예:

```text
친구들이랑 오래 할 자동화 퀘스트팩
저사양 바닐라 플러스
Create 느낌의 테크팩
RPG 탐험 던전 많은 팩
```

### 7.2 검색어 번역/전처리

검색어도 데이터셋과 같은 영어 단일 공간 처리 규칙을 사용한다.

```text
검색어가 영어:
    english_query = 원문

검색어가 한국어:
    english_query = 원문 → English 직접 번역

검색어가 기타 언어:
    english_query = 원문 → English 직접 번역
```

`english_query`는 문서와 동일한 영어 전처리 규칙을 거친다.

검색어에는 tags/categories가 없으므로 meta_tokens 삽입은 하지 않는다.

### 7.3 Word2Vec 검색어 확장

전처리된 영어 검색어 토큰 각각에 대해 Word2Vec 유사어를 추가한다.

기본값:

```python
topn = 5
```

가중치 반복 규칙:

```text
원래 검색어 토큰: 5회 반복
1위 유사어: 4회 반복
2위 유사어: 3회 반복
3위 유사어: 2회 반복
4위 유사어: 1회 반복
5위 유사어: 1회 반복
```

원래 검색어 토큰의 가중치는 가장 높은 유사어보다 낮게 두지 않는다.

Word2Vec 단어장에 없는 토큰은 무시하고 원래 토큰만 유지한다.

### 7.4 Query Vector 생성

확장된 영어 검색어 토큰을 하나의 문자열로 join한다.

```python
query_text = " ".join(expanded_query_tokens)
query_vec = tfidf_vectorizer.transform([query_text])
```

검색 시 `fit_transform` 사용은 금지한다.

### 7.5 코사인 유사도

```python
cosine_sim = linear_kernel(query_vec, tfidf_matrix)
```

`cosine_sim[0]`을 각 모드팩의 텍스트 유사도 점수로 사용한다.

### 7.6 인기도 점수

다운로드 수 기반 인기도:

```python
popularity = np.log1p(downloads)
popularity = popularity / popularity.max()
```

다운로드 수가 없는 row는 0으로 처리한다.

### 7.7 최종 점수

```python
final_score = similarity_weight * similarity_score + popularity_weight * popularity_score
```

슬라이더 값이 0~100일 때:

```python
popularity_weight = slider_value / 100
similarity_weight = 1.0 - popularity_weight
```

### 7.8 후보군 제한

추천 계산 순서:

```text
1. 전체 모드팩에 대해 코사인 유사도 계산
2. 전체 유사도 결과에 메타데이터 필터 적용
3. 필터링된 결과에서 유사도 상위 1000개 후보 추출
4. 후보 안에서 인기도 보정 final_score 계산
5. final_score 기준 정렬
6. 상위 TOP_K개 반환
```

기본값:

```python
CANDIDATE_SIZE = 1000
TOP_K = 100
```

### 7.9 필터

추천 함수는 선택적으로 다음 필터를 지원해야 한다.

```text
loader
game_version
client_side
server_side
minimum_downloads
```

필터 매칭 규칙:

```text
loader:
    CSV의 loaders 컬럼을 쉼표 기준으로 분리한 뒤, 소문자화/공백 제거 후 정확히 일치하는 값만 통과

game_version:
    CSV의 game_versions 컬럼을 쉼표 기준으로 분리한 뒤, 공백 제거 후 정확히 일치하는 값만 통과

client_side:
    required / optional / unsupported / unknown 중 하나와 정확히 일치

server_side:
    required / optional / unsupported / unknown 중 하나와 정확히 일치

minimum_downloads:
    downloads >= minimum_downloads 인 row만 통과
```

동일 필터 안에 여러 값이 들어오면 OR로 처리하고, 서로 다른 필터끼리는 AND로 처리한다.

`loader` 필터는 `categories`가 아니라 `loaders` 컬럼을 기준으로 한다.

## 8. 구현 파일 구조

권장 파일:

```text
modrinth_dataset.py
preprocessor.py
translategemma_translator.py
generate_model.py
recommend_modpacks.py
```

### 8.1 preprocessor.py

역할:

```text
modrinth_dataset.csv 읽기
description + body로 source_text 생성
HTML/Markdown 정리
언어 감지
영어가 아닌 source_text만 TranslateGemma로 영어 번역
영어 토큰화/불용어/원형화 기반 검색용 토큰화
tags/categories 메타 토큰 삽입
description 컬럼 교체
complete 컬럼 추가
modrinth_dataset_preprocessed.csv 저장
```

`preprocessor.py`에는 CSV 흐름, 텍스트 정리, 불용어, 토큰화, 메타 토큰 삽입을 둔다.
Hugging Face 모델 로드, 양자화 설정, 모델 파일 검증, 실제 generate 호출, 번역 chunk/cache 구현은 넣지 않는다.

### 8.2 translategemma_translator.py

역할:

```text
TranslateGemma 모델 저장 폴더 결정
불완전한 모델 다운로드 파일 검증
Hugging Face snapshot_download 호출
AutoProcessor 로드
공식 모델 카드 방식인 AutoModelForImageTextToText 로드
BitsAndBytesConfig 기반 로드 시점 양자화
긴 source_text를 문단/문장 단위로 분할 번역
번역 캐시 읽기/쓰기
```

### 8.3 generate_model.py

역할:

```text
전처리 CSV 로드
description 컬럼 split
Word2Vec 학습
description 컬럼으로 TF-IDF 학습
tfidf_vectorizer.pkl 저장
tfidf_matrix.npz 저장
modpack_meta.csv 저장
```

### 8.4 recommend_modpacks.py

역할:

```text
모델 로드
사용자 검색어 입력
검색어를 영어로 번역/전처리
Word2Vec 확장
TF-IDF transform
코사인 유사도 계산
다운로드 수 기반 인기도 보정
필터 적용
상위 100개 출력
```

필수 함수:

```python
load_assets()
preprocess_query(query)
expand_query_with_word2vec(tokens, model)
recommend(query, popularity_slider=20, filters=None, top_k=100)
```

## 9. 예외 처리

### 9.1 번역 실패

번역 실패 시 프로그램 전체가 중단되면 안 된다.

처리:

```text
영어 원문 → 번역 없이 원문으로 전처리 계속
비영어 원문 영어 번역 실패 → 해당 row를 complete=0으로 기록
chunk 일부만 실패 → 해당 row를 완료로 간주하지 않고 다음 실행에서 재시도
출력 토큰 한도 도달 → 출력 한도를 MAX_TRANSLATE_OUTPUT_TOKENS까지 늘려 재시도
재시도 후에도 출력 토큰 한도 도달 → chunk를 더 작은 문단/문장/절 단위로 나눠 재시도
단일 문장 길이가 HARD_MAX_TRANSLATE_INPUT_TOKENS 초과 → 해당 문장은 빈 문자열로 처리하고 로그를 남김
```

검색어 번역 실패 시에는 원문 검색어를 영어 전처리 규칙으로 처리하고, 경고 로그를 남긴다. 검색 요청 전체가 중단되면 안 된다.

### 9.2 빈 설명

`description`과 `body`가 모두 비어 있으면 row는 유지하되, 전처리된 `description`은 빈 문자열로 둔다.

### 9.3 Word2Vec OOV

검색어 토큰이 Word2Vec vocabulary에 없으면 무시한다.

원래 검색어 토큰은 유지한다.

### 9.4 TF-IDF 빈 검색어

전처리 후 검색어 토큰이 전부 사라지면 빈 결과를 반환한다.

## 10. 성능 요구사항

대상 데이터 규모:

```text
약 10,000~17,000개 모드팩
```

요구사항:

```text
TF-IDF 검색은 일반 PC에서 1초 내외로 동작해야 한다.
번역은 오래 걸릴 수 있으므로 반드시 캐시한다.
Word2Vec/TF-IDF 모델은 매 검색마다 다시 학습하지 않는다.
```

## 11. 금지사항

```text
원본 CSV의 필터용 컬럼 삭제 금지
description 외 컬럼 임의 변경 금지
문서 전체 영어/한국어 양방향 번역 금지
영어 아닌 언어 → 영어 → 한국어 2단 번역 금지
검색 시 TF-IDF fit_transform 금지
title/name을 Word2Vec 학습 description에 섞기 금지
downloads/followers를 description 텍스트에 섞기 금지
모델을 매 추천 요청마다 재학습 금지
```

## 12. 완료 기준

다음이 가능해야 한다.

```text
1. 크롤러 출력 CSV를 전처리 CSV로 변환
2. description 컬럼만 추천용 텍스트로 교체됨
3. Word2Vec 모델 생성됨
4. TF-IDF vectorizer/matrix 생성됨
5. 영어가 아닌 문서는 영어로 번역된 뒤 전처리됨
6. 사용자가 한국어로 검색하면 영어로 번역되어 검색 가능
7. Word2Vec으로 검색어 확장 가능
8. TF-IDF + 코사인 유사도로 추천 가능
9. 다운로드 수 슬라이더로 인기도 반영 가능
10. 상위 100개 추천 출력 가능
11. loaders/game_versions 등 원본 컬럼으로 필터 가능
```
