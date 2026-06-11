# SRS: Modrinth 모드팩 추천 시스템

## 1. 목적

Modrinth에서 수집한 모드팩 CSV 데이터를 기반으로, 사용자가 자연어로 원하는 모드팩 조건을 입력하면 관련 모드팩을 추천하는 프로그램을 구현한다.

추천은 다음 요소를 사용한다.

```text
전처리된 설명 텍스트
+ tags/categories 증강 토큰
+ Word2Vec 기반 검색어 확장
+ TF-IDF 벡터화
+ 코사인 유사도
+ 다운로드수 기반 인기도 보정
```

---

## 2. 입력 데이터

입력 파일:

```text
./datasets/modrinth_dataset.csv
```

입력 CSV는 기존 크롤링 코드가 생성한 파일을 사용한다.

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

`description`은 짧은 설명, `body`는 긴 상세 설명이다.

---

## 3. 출력 데이터

### 3.1 전처리 CSV

출력 파일:

```text
./datasets/modrinth_dataset_preprocessed.csv
```

요구사항:

```text
원본 CSV의 모든 기존 컬럼을 유지한다.
원본 컬럼 순서는 그대로 유지한다.
새 컬럼이 필요하면 원본 컬럼 뒤에만 추가한다.
description 컬럼은 기존 위치를 유지하되 값만 추천용 전처리 텍스트로 교체한다.
body, tags, categories, downloads, loaders 등 필터용 컬럼은 원본 그대로 유지한다.
```

즉, 전처리 후에도 기존 필터 기능에서 원본 컬럼을 그대로 쓸 수 있어야 한다.

### 3.2 모델/행렬 출력

```text
./models/modpack_word2vec.model
./models/tfidf_vectorizer.pkl
./models/tfidf_matrix.npz
./models/modpack_meta.csv
```

`modpack_meta.csv`는 추천 결과 표시용 메타데이터이다.

필수 컬럼:

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

---

## 4. 전처리 요구사항

### 4.1 원문 설명 구성

각 모드팩의 원문 설명은 다음처럼 구성한다.

```text
source_text = description + "\n\n" + body
```

단, 둘 중 하나가 비어 있으면 존재하는 텍스트만 사용한다.

최종 출력 CSV에서는 `body`는 보존하고, `description`만 전처리 결과로 교체한다.

---

### 4.2 언어 감지

`source_text`의 언어를 감지한다.

우선순위:

```text
1. 한글 비율 기반 간단 판정
2. langdetect 사용
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

---

### 4.3 번역 규칙

영어 설명과 한국어 설명을 모두 만든다.

중요 요구사항:

```text
절대 2단 번역 금지.
영어 아닌 언어 → 영어 → 한국어 같은 경로 금지.
모든 번역은 원문 source_text에서 직접 수행한다.
```

언어별 처리:

```text
원문이 영어(en):
    english_text = source_text
    korean_text = source_text → Korean 직접 번역

원문이 한국어(ko):
    english_text = source_text → English 직접 번역
    korean_text = source_text

원문이 기타 언어/unknown:
    english_text = source_text → English 직접 번역
    korean_text = source_text → Korean 직접 번역
```

금지 예시:

```text
source_text → English → Korean ❌
source_text → Korean → English ❌
```

번역은 Hugging Face의 TranslateGemma를 사용한다.

모델 ID는 코드 상단 설정값으로 분리한다.

```python
TRANSLATE_MODEL_ID = "google/translategemma-27b-it"
```

TranslateGemma는 README 기준 입력 context가 2K tokens로 작으므로, 번역 입력은 문자 수가 아니라 tokenizer/processor 기준 토큰 수로 제한한다. 안전 여유를 두기 위해 번역 입력 토큰 예산은 코드 상단 설정값으로 분리한다.

```python
MAX_TRANSLATE_INPUT_TOKENS = 1800
MAX_TRANSLATE_OUTPUT_TOKENS = 2048
```

이 값은 필요할 때 조정할 수 있어야 한다. 단, TranslateGemma의 전체 입력 context인 2K tokens를 넘지 않도록 기본값은 1800으로 둔다.

TranslateGemma는 README의 chat template 방식으로 호출한다.

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

processor = AutoProcessor.from_pretrained(TRANSLATE_MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    TRANSLATE_MODEL_ID,
    device_map="auto",
)

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

`source_lang_code`와 `target_lang_code`는 `en`, `ko`, `ja`, `zh` 같은 ISO 639-1 계열 언어 코드로 넘긴다. `English`, `Korean` 같은 자연어 이름을 TranslateGemma chat template의 언어 코드로 사용하지 않는다.

번역 입력이 `MAX_TRANSLATE_INPUT_TOKENS`를 초과하면 원문을 앞에서 잘라 쓰지 않는다. 다음 순서로 분할한다.

```text
1. source_text를 빈 줄 기준 문단 단위로 분리한다.
2. 문단을 순서대로 묶되, 묶은 chunk의 chat template 적용 후 입력 토큰 수가 MAX_TRANSLATE_INPUT_TOKENS 이하가 되도록 한다.
3. 단일 문단만으로 MAX_TRANSLATE_INPUT_TOKENS를 초과하면 해당 문단 내부를 문장 단위로 분리한다.
4. 문장 단위 chunk도 MAX_TRANSLATE_INPUT_TOKENS 이하가 되도록 묶는다.
5. 단일 문장 하나가 MAX_TRANSLATE_INPUT_TOKENS를 초과하면 해당 문장은 번역 실패로 처리하고 로그를 남긴다.
```

문단 단위 분할 로그:

```text
[번역 문단 분할] index, name, source_language, target_language, paragraph_tokens, max_tokens
```

단일 문장 초과 로그:

```text
[번역 문장 길이 초과] index, name, source_language, target_language, sentence_tokens, max_tokens
```

번역 결과는 원래 chunk 순서를 유지해 `\n\n`로 다시 합친다. 이는 번역 모델이 가능한 한 문단 문맥을 유지하도록 하기 위함이다.

번역 결과는 캐시해야 한다.

```text
./datasets/translation_cache.json
```

캐시 키는 다음을 포함해야 한다.

```text
target_language
source_language
source_text hash
TRANSLATE_MODEL_ID
MAX_TRANSLATE_INPUT_TOKENS
chunking_policy_version
```

같은 원문이라도 영어 번역과 한국어 번역은 다른 캐시 항목이어야 한다.

권장 캐시 항목 구조:

```json
{
  "<cache_key>": {
    "target_language": "Korean",
    "source_language": "English",
    "source_hash": "<sha256>",
    "model_id": "<TRANSLATE_MODEL_ID>",
    "max_translate_input_tokens": 1800,
    "source_chars": 4200,
    "chunk_count": 3,
    "split_by_sentence": true,
    "skipped_overlong_segments": 0,
    "translated_text": "..."
  }
}
```

---

### 4.4 영어 전처리

영어 텍스트는 다음 규칙으로 전처리한다.

```text
소문자화
URL 제거
마크다운 잔여 문법 제거
특수문자 제거
영문, 숫자, 한글, +, #, _, -, . 정도만 보존
공백 기준 토큰화
영어 불용어 제거
길이 1 이하 일반 토큰 제거
숫자는 보존 가능
```

불용어 예시:

```text
the, a, an, and, or, to, of, in, on, for,
with, by, from, this, that, is, are,
minecraft, modpack, modpacks, mod, mods
```

---

### 4.5 한국어 전처리

한국어 텍스트는 Okt를 사용한다.

```python
okt.pos(text, norm=True, stem=True)
```

남길 품사:

```text
Noun
Verb
Adjective
Adverb
Alpha
Number
Foreign
```

버릴 품사:

```text
Josa
Eomi
PreEomi
Punctuation
KoreanParticle
Exclamation
Determiner
Suffix
```

한국어 불용어 예시:

```text
것, 수, 등, 때, 곳, 거, 점, 중, 전, 후,
내, 외, 위, 뒤, 더, 또, 및,
마인크래프트, 마크, 모드팩, 팩, 모드
```

길이 1 이하 일반 토큰은 제거한다. 숫자는 보존 가능하다.

---

### 4.6 tags/categories 증강

`tags`, `categories` 컬럼은 쉼표 기준으로 파싱한다.

예:

```text
tags = "Kitchen Sink, Optimization"
categories = "kitchen-sink, optimization"
```

각 항목에 대해 다음 토큰을 만든다.

```text
원본 전처리 토큰
prefix 토큰
```

예:

```text
Kitchen Sink
→ kitchen sink
→ TAG_kitchen_sink

optimization
→ optimization
→ CAT_optimization
```

prefix 규칙:

```text
tags 항목       → TAG_xxx
categories 항목 → CAT_xxx
```

정규화 규칙:

```text
소문자화
공백/특수문자 → _
연속된 _ 축약
앞뒤 _ 제거
```

---

### 4.7 tags/categories 삽입 규칙

tags/categories 메타 토큰은 영어 토큰열과 한국어 토큰열 각각에 독립적으로 삽입한다.

기본 삽입 주기:

```python
META_INTERVAL = 20
```

최종 구조:

```text
[meta_tokens]
영어 토큰 20개
[meta_tokens]
영어 토큰 20개
[meta_tokens]
...
[meta_tokens]

[meta_tokens]   # 영어/한국어 경계 강제 삽입

[meta_tokens]
한국어 토큰 20개
[meta_tokens]
한국어 토큰 20개
[meta_tokens]
...
[meta_tokens]
```

요구사항:

```text
각 언어 토큰열의 시작에 meta_tokens 삽입
각 언어 토큰열의 시작으로부터 20, 40, 60...번째 토큰 뒤에 meta_tokens 삽입
각 언어 토큰열의 끝에 meta_tokens 삽입
영어/한국어 설명 경계에는 별도 meta_tokens 삽입
```

영한 경계 삽입은 별도 신호이므로, 영어 끝 삽입이나 한국어 시작 삽입과 인접해 `meta_tokens`가 연속으로 나타날 수 있다.

즉 짧은 설명이어도 최소한 다음 구조가 될 수 있다.

```text
meta + english_tokens + meta + meta + korean_tokens + meta
```

---

### 4.8 최종 description 생성

전처리된 최종 텍스트는 다음으로 구성한다.

```text
english_tokens_with_meta + korean_tokens_with_meta
```

이 문자열을 원본 CSV의 `description` 컬럼에 덮어쓴다.

금지:

```text
name/title을 description에 섞지 않는다.
downloads/followers를 description에 섞지 않는다.
loaders/game_versions는 description에 섞지 않는다.
```

이유:

```text
name은 고유명사가 많아 Word2Vec/TF-IDF에 노이즈가 될 수 있음.
downloads/followers는 별도 점수 계산에 사용해야 함.
loaders/game_versions는 추천 후 필터에 사용해야 함.
```

---

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

---

### 5.2 학습 데이터

학습 입력:

```text
전처리 CSV의 description 컬럼
```

각 description을 공백 기준으로 split하여 토큰 리스트로 사용한다.

예:

```python
sentences = df["description"].fillna("").apply(lambda x: x.split()).tolist()
```

---

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

설명:

```text
sg=1: Skip-gram 사용
window=10: tags/categories 삽입 토큰과 설명 토큰의 관계를 학습하기 위함
min_count=2: 1회성 노이즈 토큰 제거
```

주의:

```text
name/title은 Word2Vec 학습에 넣지 않는다.
희귀 고유명사가 너무 많아지면 most_similar 품질이 깨진다.
```

---

### 5.4 저장

```text
./models/modpack_word2vec.model
```

저장 후 추천 단계에서 로드 가능해야 한다.

---

## 6. TF-IDF 요구사항

### 6.1 목적

TF-IDF는 모든 모드팩 설명을 숫자 벡터 행렬로 변환한다.

추천 시 사용자의 검색어도 같은 vectorizer로 transform하여 코사인 유사도를 계산한다.

---

### 6.2 학습 데이터

입력:

```text
전처리 CSV의 description 컬럼
```

---

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

---

### 6.4 저장

```text
./models/tfidf_vectorizer.pkl
./models/tfidf_matrix.npz
```

`sparse matrix`로 저장한다.

```python
scipy.sparse.save_npz(...)
```

---

## 7. 추천 요구사항

### 7.1 검색 입력

사용자가 자연어 검색어를 입력한다.

예:

```text
친구들이랑 오래 할 자동화 퀘스트팩
저사양 바닐라 플러스
Create 느낌의 테크팩
RPG 탐험 던전 많은 팩
```

---

### 7.2 검색어 번역

검색어도 영어/한국어 양쪽 토큰을 만든다.

규칙:

```text
검색어가 영어:
    english_query = 원문
    korean_query = 원문 → Korean 직접 번역

검색어가 한국어:
    english_query = 원문 → English 직접 번역
    korean_query = 원문

검색어가 기타 언어:
    english_query = 원문 → English 직접 번역
    korean_query = 원문 → Korean 직접 번역
```

마찬가지로 2단 번역 금지.

---

### 7.3 검색어 전처리

검색어도 데이터셋과 같은 방식으로 전처리한다.

```text
english_query → 영어 전처리
korean_query → 한국어 형태소 전처리
```

단, 검색어에는 tags/categories가 없으므로 meta_tokens 삽입은 하지 않는다.

---

### 7.4 Word2Vec 검색어 확장

검색어 토큰 각각에 대해 Word2Vec 유사어를 추가한다.

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

원래 검색어 토큰의 가중치는 가장 높은 유사어보다 낮게 두지 않는다. 반복 횟수는 TF-IDF query vector에서 해당 토큰의 영향력을 키우는 간단한 가중치 역할을 하며, 유사어가 원문보다 강하면 사용자가 입력한 의도가 희석될 수 있다.

Word2Vec 단어장에 없는 토큰은 무시하고 원래 토큰만 유지한다.

예외 발생 금지:

```python
try:
    similar = model.wv.most_similar(token, topn=5)
except KeyError:
    pass
```

---

### 7.5 Query Vector 생성

확장된 검색어 토큰을 하나의 문자열로 join한다.

```python
query_text = " ".join(expanded_query_tokens)
```

그리고 기존 TF-IDF vectorizer로 transform한다.

```python
query_vec = tfidf_vectorizer.transform([query_text])
```

금지:

```python
fit_transform 사용 금지
```

---

### 7.6 코사인 유사도 계산

```python
cosine_sim = linear_kernel(query_vec, tfidf_matrix)
```

`cosine_sim[0]`을 각 모드팩의 텍스트 유사도 점수로 사용한다.

---

### 7.7 인기도 점수

다운로드 수를 사용해 인기도 점수를 만든다.

```python
popularity = np.log1p(downloads)
popularity = popularity / popularity.max()
```

다운로드 수가 없는 경우 0으로 처리한다.

---

### 7.8 최종 점수

추천 결과는 유사도와 인기도를 섞어 계산한다.

```python
final_score = similarity_weight * similarity_score + popularity_weight * popularity_score
```

슬라이더 값이 0~100일 때:

```python
popularity_weight = slider_value / 100
similarity_weight = 1.0 - popularity_weight
```

예:

```text
slider=0   → 유사도 100%
slider=20  → 유사도 80%, 인기도 20%
slider=50  → 유사도 50%, 인기도 50%
slider=100 → 인기도 100%
```

단, 완전 인기순 도배를 막기 위해 후보군 제한을 적용한다.

---

### 7.9 후보군 제한

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

이유:

```text
처음부터 상위 100개만 자르면 인기도 슬라이더가 힘을 못 씀.
전체를 인기도로 섞으면 관련 없는 인기팩이 튀어나올 수 있음.
상위 1000개 후보 후 재정렬이 가장 안전함.
```

---

### 7.10 필터

추천 함수는 선택적으로 다음 필터를 지원해야 한다.

```text
loader
game_version
client_side
server_side
minimum_downloads
```

필터는 전체 유사도 계산 후, 후보군 1000개를 자르기 전에 적용한다.

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

Modrinth 검색 API에서는 loader가 `categories` facet에 포함되지만, 크롤러는 project 상세 응답의 `loaders` 필드를 별도 컬럼으로 저장한다. 추천 단계의 loader 필터는 `categories`가 아니라 `loaders` 컬럼을 기준으로 한다.

`game_versions` 필터 정확도를 보장하려면 크롤러는 project 상세 응답의 전체 `game_versions`를 저장해야 한다. 최신 5개만 저장하면 필터도 그 5개 버전 범위에서만 정확하다.

권장 순서:

```text
1. 전체 유사도 계산
2. 메타데이터 필터 적용
3. 후보군 1000개 추출
4. 인기도 보정
5. 상위 TOP_K개 반환
```

---

## 8. 구현 파일 구조

권장 파일:

```text
preprocessor.py
generate_model.py
recommend_modpacks.py
```

또는 단일 파일로 시작해도 되지만, 함수는 분리해야 한다.

---

### 8.1 `preprocessor.py`

역할:

```text
modrinth_dataset.csv 읽기
description + body로 source_text 생성
언어 감지
Hugging Face TranslateGemma로 영어/한국어 설명 생성
긴 source_text는 문단 단위로 나누고, 단일 문단이 토큰 예산을 넘으면 문장 단위로 나누어 번역
영어/한국어 전처리
tags/categories 증강 토큰 삽입
description 컬럼 교체
modrinth_dataset_preprocessed.csv 저장
```

필수 함수:

```python
detect_language(text)
translate_text(text, source_lang, target_lang)
cached_translate(text, source_lang, target_lang, cache)
split_text_for_translation(text, source_lang, target_lang)
preprocess_english(text)
preprocess_korean(text)
build_meta_tokens(tags, categories)
insert_meta(tokens, meta_tokens, interval=20)
build_index_description(en_tokens, ko_tokens, meta_tokens)
```

---

### 8.2 `generate_model.py`

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

---

### 8.3 `recommend_modpacks.py`

역할:

```text
모델 로드
사용자 검색어 입력
검색어 번역/전처리
Word2Vec 확장
TF-IDF transform
코사인 유사도 계산
다운로드수 기반 인기도 보정
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

---

## 9. 예외 처리 요구사항

### 9.1 번역 실패

번역 실패 시 프로그램 전체가 중단되면 안 된다.

처리:

```text
영어 번역 실패 → 원문을 english_text로 사용
한국어 번역 실패 → korean_text를 빈 문자열로 사용
chunk 일부만 실패 → 실패한 chunk는 빈 문자열로 처리하고 나머지 chunk 번역 결과는 유지
단일 문장 길이 초과 → 해당 문장은 빈 문자열로 처리하고 로그를 남김
```

로그 출력:

```text
[번역 실패] index, name, source_language, target_language, error
```

---

### 9.2 빈 설명

`description`과 `body`가 모두 비어 있으면 해당 row는 유지하되, 전처리된 `description`은 빈 문자열로 둔다.

추천 단계에서는 빈 description row의 유사도는 자연스럽게 0이 된다.

참고: 기본 크롤러가 `description`과 `body`가 모두 있는 row만 생성하도록 구현되어 있다면 빈 설명 row는 일반적으로 발생하지 않는다. 이 조항은 외부 CSV를 입력으로 받거나 수집 정책이 바뀌는 경우를 위한 방어 규칙이다.

---

### 9.3 Word2Vec OOV

검색어 토큰이 Word2Vec vocabulary에 없으면 무시한다.

원래 검색어 토큰은 유지한다.

---

### 9.4 TF-IDF 빈 검색어

전처리 후 검색어 토큰이 전부 사라지면 빈 결과를 반환한다.

```text
검색어에서 유효 토큰을 찾지 못했습니다.
```

---

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

---

## 11. 금지사항

```text
원본 CSV의 필터용 컬럼 삭제 금지
description 외 컬럼 임의 변경 금지
영어 아닌 언어 → 영어 → 한국어 2단 번역 금지
검색 시 TF-IDF fit_transform 금지
title/name을 Word2Vec 학습 description에 섞기 금지
downloads/followers를 description 텍스트에 섞기 금지
모델을 매 추천 요청마다 재학습 금지
```

---

## 12. 완료 기준

다음이 가능해야 한다.

```text
1. 크롤러 출력 CSV를 전처리 CSV로 변환
2. description 컬럼만 추천용 텍스트로 교체됨
3. Word2Vec 모델 생성됨
4. TF-IDF vectorizer/matrix 생성됨
5. 사용자가 한국어로 검색 가능
6. Word2Vec으로 검색어 확장 가능
7. TF-IDF + 코사인 유사도로 추천 가능
8. 다운로드수 슬라이더로 인기도 반영 가능
9. 상위 100개 추천 출력 가능
10. loaders/game_versions 등 원본 컬럼으로 필터 가능
```

---

## 13. Codex 작업 지시 요약

```text
위 SRS에 맞춰 기존 modrinth_dataset.py 결과물인 ./datasets/modrinth_dataset.csv를 입력으로 사용하는 전처리/학습/추천 파이프라인을 구현하라.

원본 CSV의 description 컬럼만 추천용 전처리 텍스트로 교체하고, 나머지 컬럼은 필터용으로 보존하라.

description + body를 source_text로 사용하되, 출력 CSV의 body는 원본 그대로 유지하라.

언어 감지 후 영어 설명과 한국어 설명을 만들되, 절대 영어를 중간 경유하는 2단 번역을 하지 마라. Hugging Face TranslateGemma chat template을 사용하고, 원문 source_text에서 en, ko로 각각 직접 번역하라.

TranslateGemma 입력은 2K tokens context 한계를 고려해 토큰 수 기준으로 제한하라. source_text가 길면 문단 단위로 chunk를 만들고, 단일 문단이 토큰 예산을 넘으면 문장 단위로 나누어 번역하라. 앞부분만 잘라 번역하는 방식은 금지한다.

전처리된 영어/한국어 토큰 사이에 tags/categories 기반 meta_tokens를 20토큰마다 삽입하라. 시작, 끝, 영한 경계에는 20토큰 미달이어도 meta_tokens를 강제로 삽입하라.

Word2Vec은 description 컬럼 토큰으로 학습하고, 검색어 확장용으로 사용하라.

TF-IDF는 description 컬럼으로 학습하고, 검색 시 query는 transform만 사용하라.

추천은 TF-IDF 코사인 유사도 + log1p(downloads) 인기도 점수를 슬라이더로 섞어 계산하라. 전체 유사도 계산 후 필터를 적용하고, 필터링된 결과의 상위 1000개 후보 안에서 재정렬한 뒤 최종 상위 100개를 반환하라.
```
