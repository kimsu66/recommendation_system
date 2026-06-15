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

모델 생성 단계는 다음 산출물을 만든다.

```text
Word2Vec 모델
TF-IDF vectorizer
TF-IDF matrix
```

추천 결과 표시, 필터, 인기도 계산에 필요한 메타데이터는 전처리 CSV의 원본 컬럼을 그대로 사용한다.
별도 메타데이터 CSV는 필수 산출물로 두지 않는다.

## 4. 전처리 요구사항

### 4.1 원문 설명 구성

각 row의 전처리 원문은 다음과 같이 만든다.

```text
source_text = description + "\n\n" + body
```

둘 중 하나가 비어 있으면 존재하는 텍스트만 사용한다.

출력 CSV에서는 `body`를 변경하지 않고, `description`만 전처리 결과로 교체한다.

### 4.2 HTML/Markdown 정리

번역 또는 토큰화 전에 `source_text`에서 HTML/Markdown/링크 노이즈를 정리한다.

요구사항:

```text
HTML block 태그는 줄바꿈 또는 문단 경계로 변환한다.
HTML inline 태그는 텍스트만 보존한다.
script/style/iframe/svg/video/audio/picture/img/pre/code/kbd/samp는 제거한다.
Markdown 이미지는 제거한다.
Markdown 링크는 URL을 제거하고 링크 텍스트만 보존한다.
raw URL은 제거한다.
코드 블록과 inline code는 제거한다.
이모지와 장식용 특수 기호는 제거한다.
표 구분선과 장식용 구분선은 제거한다.
목록 항목은 줄 단위 텍스트로 보존한다.
```

### 4.3 언어 감지

`source_text`의 언어를 감지한다.

우선순위:

```text
1. 한글/가나/CJK 문자권 힌트로 ko/ja/zh 보정
2. Lingua 언어 식별기로 주 언어 판단
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

### 4.4 언어 정규화

최종 추천 인덱스는 영어 단일 텍스트 공간으로 만든다.

문서와 검색어는 같은 언어 공간에서 비교되어야 한다.
따라서 영어가 아닌 입력은 필요한 경우에만 영어로 정규화한다.

```text
영어 입력은 번역하지 않는다.
영어가 아닌 문서 원문은 영어로 직접 정규화한다.
영어가 아닌 검색어는 검색 시점에 영어로 직접 정규화한다.
영어/한국어 양방향 전체 번역 결과를 함께 저장하지 않는다.
영어 아닌 언어 → 영어 → 한국어 같은 2단 번역은 금지한다.
언어 감지 실패 시에는 원문을 보존하되 완료 처리 기준을 흐리지 않는다.
```

구현 방향:
번역 백엔드는 교체 가능하게 두고, 모델명/양자화/입력 형식 같은 세부 설정은 코드 설정값으로 관리한다.
SRS는 특정 번역 모델 호출 방식에 묶이지 않는다.

### 4.5 긴 입력 처리

번역이 필요한 긴 설명은 번역 백엔드 입력 한도를 넘지 않도록 나누어 처리한다.
문자 수만으로 자르지 말고, 가능한 경우 tokenizer/processor 기준 길이를 사용한다.

문단/문장 chunk 절차:

```text
1. source_text를 빈 줄 기준 문단 단위로 분리한다.
2. 문단을 순서대로 묶어 번역 입력 한도 이하의 chunk를 만든다.
3. 단일 문단이 입력 한도를 넘으면 문장 단위로 나눈다.
4. 단일 문장도 입력 한도를 넘으면 더 작은 단위로 나누거나 해당 chunk를 실패 처리한다.
5. chunk 번역 결과는 원래 순서를 유지해 다시 합친다.
```

앞부분만 잘라 번역하고 완료 처리하는 방식은 금지한다.

### 4.6 처리 상태와 번역 캐시

전처리 중단 후 재개는 최종 출력 CSV의 `complete` 컬럼으로 판단한다.

```text
complete
```

`complete=1`이면 전처리 완료 row로 본다.
`complete=0`, 빈 값, 컬럼 없음은 미완료로 보고 재실행 시 다시 전처리한다.
row 처리가 끝날 때마다 최종 출력 CSV에 append한다.

```text
정상 완료 row: complete=1
처리 실패 row: complete=0
```

번역 결과는 같은 입력을 반복 번역하지 않도록 캐시할 수 있다.
캐시 키 구조와 저장 방식은 구현에서 정하되, 번역 모델이나 설정이 바뀌었을 때 오래된 캐시를 잘못 재사용하지 않아야 한다.

전처리 종료 시 이번 실행에서 새로 성공한 row 수와 실패한 row 수를 출력한다.

### 4.7 텍스트 전처리

최종 `description`에 들어갈 텍스트는 영어 기준 전처리를 거친 공백 기준 토큰 문자열로 만든다.

기본 규칙:

```text
소문자화
URL 제거
마크다운 잔여 문법 제거
영문, 숫자, +, #, _, -, . 정도만 보존
문서별로 다르게 적힌 주요 다단어 표현을 단일 토큰으로 정규화
영어 형태소/토큰 분리
불용어 제거
검색 의미가 약한 일반어 제거
spaCy 기반 품사 판정과 원형 변환(lemmatization)
모드명/로더명/주요 도메인 토큰/메타 토큰 보존
한 글자 토큰 제거
숫자만 있는 토큰 제거
모드팩 자체 버전 또는 changelog 버전 제거
Minecraft 게임 버전은 추천 텍스트에서 제거하고 game_versions 필터에서만 사용
```

문서와 검색어는 모두 같은 영어 전처리 규칙을 사용한다. 한국어 형태소 분석은 최종 인덱스가 영어 단일 공간이므로 기본 요구사항에서 제외한다.

다단어 정규화 예:

```text
data pack, data-pack → datapack
resource pack → resourcepack
world generation → worldgen
vanilla plus, vanilla+ → vanillaplus
voice chat → voicechat
Farmer's Delight → farmersdelight
```

버전/숫자 처리 예:

```text
v2.0~2.2, v1.x 같은 문서/changelog 버전 표기는 제거
1.20.1, 1.21 같은 Minecraft 버전은 description 토큰에서 제거
2, 20, 2024 같은 숫자 전용 토큰은 제거
ae2, c2me, 3d처럼 숫자와 문자가 섞인 의미 토큰은 전처리 규칙에 따라 보존 가능
```

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
영어 토큰열의 시작으로부터 30, 60, 90...번째 토큰 뒤에 meta_tokens 삽입
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

구현 방향:
참고 프로젝트의 `job05_word2vec.py`처럼 전처리된 `description`을 공백 기준 토큰 리스트로 만든 뒤 Word2Vec을 학습한다.
벡터 크기, window, min_count, epochs 같은 값은 코드 상단 설정값으로 둔다.

주의:

```text
name/title은 Word2Vec 학습에 넣지 않는다.
희귀 고유명사가 너무 많아지면 most_similar 품질이 깨질 수 있다.
```

### 5.4 저장

```text
models/ 아래 Word2Vec 모델 파일
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

구현 방향:
참고 프로젝트의 `job03_TFIDF.py`처럼 전처리된 `description` 컬럼으로 TF-IDF vectorizer와 matrix를 만든다.

요구사항:

```text
이미 전처리된 공백 분리 토큰 문자열을 입력으로 사용
fit_transform은 학습 시에만 사용
검색어에는 transform만 사용
```

### 6.4 저장

```text
models/ 아래 TF-IDF vectorizer 파일
models/ 아래 TF-IDF matrix 파일
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

구현 방향:
참고 프로젝트의 keyword 기반 추천처럼 원래 검색어 토큰에 높은 가중치를 두고, Word2Vec 유사어를 낮은 가중치로 추가한다.
유사어 개수와 가중치 방식은 설정값으로 조정 가능하게 둔다.

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

## 8. 구현 구조 방향

큰 흐름은 참고 프로젝트 `movie_review_word2vac`처럼 단계별 job 스크립트 구조를 따른다.
파일명은 구현 중 조정할 수 있지만, 단계 경계는 다음처럼 유지한다.

```text
데이터 수집 단계
전처리 단계
TF-IDF 모델 생성 단계
Word2Vec 모델 생성 단계
추천 로직 확인 단계
GUI 또는 실행 앱 단계
```

### 8.1 데이터 수집 단계

역할:

```text
Modrinth 원본 데이터 수집
필수 컬럼을 가진 원본 CSV 생성
```

### 8.2 전처리 단계

역할:

```text
원본 CSV 로드
description + body로 source_text 생성
HTML/Markdown 정리
언어 감지
필요한 경우에만 영어 정규화 수행
spaCy 기반 영어 토큰화/품사/원형화와 불용어 제거
tags/categories 메타 토큰 삽입
description 컬럼 교체
complete 컬럼으로 row 처리 상태 기록
전처리 CSV 저장
```

구현 방향:
CSV 흐름, 텍스트 정리, 불용어, spaCy 기반 영어 전처리, 메타 토큰 삽입은 전처리 단계에 둔다.
번역 백엔드 호출과 긴 입력 분할은 별도 helper/module로 분리해 전처리 흐름이 과하게 복잡해지지 않게 한다.

### 8.3 TF-IDF 모델 생성 단계

역할:

```text
전처리 CSV 로드
description 컬럼으로 TF-IDF vectorizer 학습
TF-IDF matrix 생성
모델 산출물 저장
```

### 8.4 Word2Vec 모델 생성 단계

역할:

```text
전처리 CSV 로드
description 컬럼 split
Word2Vec 학습
모델 산출물 저장
```

### 8.5 추천 로직 확인 단계

역할:

```text
전처리 CSV와 모델 산출물 로드
사용자 검색어 입력
필요한 경우 검색어 영어 정규화
검색어 전처리
Word2Vec 검색어 확장
TF-IDF transform
코사인 유사도 계산
다운로드 수 기반 인기도 보정
필터 적용
상위 결과 출력
```

구현 방향:
추천 점수식과 필터 규칙은 SRS를 따른다.
전처리 CSV의 원본 컬럼을 결과 표시, 필터, 인기도 계산에 재사용하고, 별도 메타 CSV 생성을 기본 요구사항으로 두지 않는다.

### 8.6 GUI 또는 실행 앱 단계

역할:

```text
검색어 입력
필터/인기도 슬라이더 입력
추천 결과 표시
```

GUI는 참고 프로젝트처럼 마지막 실행 계층으로 두고, 전처리/모델 생성/추천 계산 로직과 강하게 결합하지 않는다.

## 9. 예외 처리

### 9.1 번역 실패

문서 또는 검색어 정규화 중 번역이 실패해도 프로그램 전체가 중단되면 안 된다.

처리:

```text
영어 원문 → 번역 없이 원문으로 전처리 계속
비영어 원문 영어 번역 실패 → 해당 row를 complete=0으로 기록
chunk 일부만 실패 → 해당 row를 완료로 간주하지 않고 다음 실행에서 재시도
긴 입력 처리 실패 → 문단/문장/절 단위로 더 나누어 재시도
그래도 실패한 chunk가 있으면 해당 row를 완료로 기록하지 않음
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
번역이 필요한 경우 반복 번역을 줄이기 위해 캐시를 사용할 수 있다.
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
인기도/필터 계산을 위해 원본 메타데이터를 불필요하게 중복 저장하지 않기
```

## 12. 완료 기준

다음이 가능해야 한다.

```text
1. 크롤러 출력 CSV를 전처리 CSV로 변환
2. description 컬럼만 추천용 텍스트로 교체됨
3. Word2Vec 모델 생성됨
4. TF-IDF vectorizer/matrix 생성됨
5. 영어가 아닌 문서는 필요한 경우 영어 정규화를 거쳐 전처리됨
6. 사용자가 한국어 등 비영어로 검색하면 필요한 경우 영어 정규화를 거쳐 검색 가능
7. Word2Vec으로 검색어 확장 가능
8. TF-IDF + 코사인 유사도로 추천 가능
9. 다운로드 수 슬라이더로 인기도 반영 가능
10. 상위 100개 추천 출력 가능
11. loaders/game_versions 등 원본 컬럼으로 필터 가능
```
