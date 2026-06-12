"""Modrinth modpack CSV preprocessor.

이 파일이 하는 일:
- `INPUT_CSV`에서 크롤러가 만든 원본 CSV를 읽는다.
- 각 row의 `description + body`를 원문 설명으로 보고, 영어/한국어 검색용 텍스트를 만든다.
- tags/categories를 메타 토큰으로 만들어 검색용 `description` 문자열 안에 삽입한다.
- 출력 CSV에는 원본 컬럼 순서를 유지하고, `description` 컬럼 값만 전처리 결과로 교체한다.

이 파일에 넣으면 안 되는 일:
- Word2Vec/TF-IDF 학습은 여기서 하지 않는다. 그 작업은 `generate_model.py`에서 한다.
- 추천/필터/랭킹도 여기서 하지 않는다. 그 작업은 추천 모듈에서 한다.
"""

import hashlib
import json
import os
import re

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        """tqdm이 없을 때 진행바 없이 그대로 반복하기 위한 fallback이다."""
        return iterable


# =========================
# 경로 설정
# =========================

# 크롤러가 만든 원본 CSV 경로를 넣는다.
INPUT_CSV = "./datasets/modrinth_dataset.csv"
# 이 스크립트가 저장할 전처리 CSV 경로다. description 컬럼만 교체된다.
OUTPUT_CSV = "./datasets/modrinth_dataset_preprocessed.csv"
# TranslateGemma 결과 캐시 파일이다. 같은 설명을 매번 다시 번역하지 않기 위해 쓴다.
TRANSLATION_CACHE = "./datasets/translation_cache.json"

# Hugging Face에서 받을 TranslateGemma 모델 ID를 넣는다.
TRANSLATE_MODEL_ID = "google/translategemma-27b-it"
# 번역 없이 파이프라인만 확인하고 싶으면 False로 바꾼다.
USE_TRANSLATION = True

# tags/categories 메타 토큰을 설명 토큰 몇 개마다 다시 끼워 넣을지 정한다.
META_INTERVAL = 20
# TranslateGemma README의 2K input context보다 낮게 잡은 안전 예산이다.
MAX_TRANSLATE_INPUT_TOKENS = 1800
# chunk 하나를 번역할 때 생성할 최대 출력 토큰 수다.
MAX_TRANSLATE_OUTPUT_TOKENS = 2048
# 캐시 무효화용 정책 버전이다. chunk 분할 방식을 바꾸면 이 값을 바꾼다.
CHUNKING_POLICY_VERSION = "paragraph-then-sentence-v1"


# =========================
# 품사 / 불용어
# =========================

KEEP_KO_POS = {
    "Noun",
    "Verb",
    "Adjective",
    "Adverb",
    "Alpha",
    "Number",
    "Foreign",
}

# 한국어 번역 결과에서 검색 의미가 약한 일반 표현을 제거한다.
# 장르/의도 단어(퀘스트, 탐험, 자동화, 최적화 등)는 여기에 넣지 않는다.
KO_STOPWORDS = {
    "것", "수", "등", "때", "곳", "거", "점", "중", "전", "후",
    "내", "외", "위", "뒤", "더", "또", "및", "좀", "매우",
    "마인크래프트", "마크", "모드팩", "팩", "모드",
    "공식", "버전", "포함", "추가", "제공", "사용", "플레이",
    "경험", "패키지", "구성", "기반", "중심", "위한", "통해",
    "당신", "여러", "모든", "많은", "새로운", "간단하다",
    "패브릭", "포지", "네오포지", "퀼트",
}

# Modrinth 실제 모드팩 설명에서 자주 반복되는 보일러플레이트를 제거한다.
# `create`, `vanilla`, `quest`, `performance`, `optimization`처럼 검색 의도인 단어는 보존한다.
EN_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for",
    "with", "by", "from", "this", "that", "these", "those",
    "is", "are", "was", "were", "be", "been", "as", "at",
    "it", "its", "you", "your", "into", "will", "can",
    "minecraft", "modpack", "modpacks", "mod", "mods",
    "pack", "packs", "package", "packages",
    "include", "includes", "included", "including",
    "feature", "features", "content", "experience",
    "official", "version", "new", "more", "all", "everything",
    "need", "needs", "made", "built", "designed", "based", "focused",
    "simple", "proper", "true", "best", "popular",
    "play", "playing", "game", "mc",
    "fabric", "forge", "neoforge", "quilt",
}

# 입력 CSV에 반드시 있어야 하는 컬럼이다.
# 이 스크립트는 새 컬럼을 요구하지 않고, 아래 기존 컬럼만 보존한다.
REQUIRED_COLUMNS = [
    "name",
    "slug",
    "url",
    "description",
    "body",
    "tags",
    "categories",
    "loaders",
    "game_versions",
    "client_side",
    "server_side",
    "license",
    "downloads",
    "followers",
    "date_created",
    "date_modified",
]

# langdetect 결과나 내부 코드명을 TranslateGemma chat template용 언어 코드로 정규화한다.
LANG_CODE_ALIASES = {
    "en": "en",
    "ko": "ko",
    "kr": "ko",
    "ja": "ja",
    "jp": "ja",
    "zh": "zh",
    "zh-cn": "zh",
    "zh-tw": "zh",
    "zh-hans": "zh",
    "zh-hant": "zh",
}

_okt = None


def get_okt():
    """Okt 형태소 분석기를 필요할 때만 로드한다.

    여기에 따로 넣을 값은 없다. 실행 전에 `konlpy`와 Java 환경이 준비되어 있어야 한다.
    설치되어 있지 않으면 한국어 전처리 단계에서 명확한 오류를 낸다.
    """
    global _okt

    if _okt is not None:
        return _okt

    try:
        from konlpy.tag import Okt
    except ImportError as e:
        raise ImportError(
            "konlpy가 설치되어 있지 않습니다. 한국어 전처리를 실행하려면 "
            "`pip install konlpy`와 Java 런타임 설치가 필요합니다."
        ) from e

    _okt = Okt()
    return _okt


# =========================
# 번역 모델
# =========================

_model = None
_processor = None


def load_translator():
    """TranslateGemma 모델과 processor를 한 번만 로드한다.

    여기에 들어가는 값:
    - `TRANSLATE_MODEL_ID`: Hugging Face 모델 ID

    반환값:
    - model: 실제 번역을 수행하는 Hugging Face 모델
    - processor: TranslateGemma chat template과 tokenizer를 제공하는 processor
    """
    global _model, _processor

    if not USE_TRANSLATION:
        return None, None

    if _model is not None:
        return _model, _processor

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    _processor = AutoProcessor.from_pretrained(TRANSLATE_MODEL_ID)
    _model = AutoModelForImageTextToText.from_pretrained(
        TRANSLATE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    return _model, _processor


def normalize_lang_code(lang: str) -> str:
    """언어 감지 결과를 TranslateGemma가 받는 언어 코드로 바꾼다."""
    lang = str(lang or "").strip().lower().replace("_", "-")
    if not lang or lang == "unknown":
        return "unknown"
    if lang in LANG_CODE_ALIASES:
        return LANG_CODE_ALIASES[lang]
    return lang.split("-", 1)[0]


def build_translation_messages(text: str, source_lang: str, target_lang: str) -> list[dict]:
    """TranslateGemma README 형식의 chat template 입력을 만든다.

    여기에 넣는 값:
    - text: 번역할 원문 chunk
    - source_lang: 원문 언어 코드(en, ko, ja, zh 등)
    - target_lang: 목표 언어 코드(en 또는 ko)
    """
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": source_lang,
                    "target_lang_code": target_lang,
                    "text": text,
                }
            ],
        }
    ]


def get_model_device(model):
    """모델이 올라간 장치를 찾아 입력 tensor를 같은 장치로 보낼 때 사용한다."""
    device = getattr(model, "device", None)
    if device is not None:
        return device
    return next(model.parameters()).device


def count_translation_input_tokens(text: str, source_lang: str, target_lang: str, processor) -> int:
    """TranslateGemma chat template 적용 후 실제 입력 토큰 수를 계산한다.

    문자 수가 아니라 모델 입력 토큰 수를 기준으로 chunk를 나누기 위해 필요하다.
    """
    messages = build_translation_messages(text, source_lang, target_lang)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return len(inputs["input_ids"][0])


def split_paragraphs(text: str) -> list[str]:
    """빈 줄 기준으로 문단을 나눈다.

    번역 품질을 위해 가능한 한 문단 단위 문맥을 유지한다.
    """
    return [part.strip() for part in re.split(r"\n\s*\n+", str(text).strip()) if part.strip()]


def split_sentences(paragraph: str) -> list[str]:
    """단일 문단이 너무 길 때만 문장 단위로 나눈다."""
    paragraph = str(paragraph).strip()
    if not paragraph:
        return []

    pieces = re.split(r"(?<=[.!?。！？])\s+", paragraph)
    pieces = [piece.strip() for piece in pieces if piece.strip()]
    return pieces if pieces else [paragraph]


def pack_segments_by_token_budget(
    segments: list[str],
    separator: str,
    source_lang: str,
    target_lang: str,
    processor,
    row_index=None,
    name: str = "",
    segment_kind: str = "paragraph",
) -> tuple[list[str], int, bool]:
    """문단 또는 문장을 토큰 예산 이하의 번역 chunk로 묶는다.

    여기에 넣는 값:
    - segments: 문단 목록 또는 문장 목록
    - separator: chunk 안에서 segment를 다시 합칠 때 쓸 구분자
    - source_lang/target_lang: TranslateGemma 언어 코드

    반환값:
    - chunks: 실제 번역할 문자열 목록
    - skipped: 단일 문장도 너무 길어서 건너뛴 개수
    - split_by_sentence: 문장 단위 fallback이 발생했는지 여부
    """
    chunks = []
    current = ""
    skipped = 0
    split_by_sentence = False

    for segment in segments:
        segment_tokens = count_translation_input_tokens(segment, source_lang, target_lang, processor)

        if segment_tokens > MAX_TRANSLATE_INPUT_TOKENS:
            if segment_kind == "sentence":
                print(
                    "[번역 문장 길이 초과] "
                    f"index={row_index}, name={name}, source_language={source_lang}, "
                    f"target_language={target_lang}, sentence_tokens={segment_tokens}, "
                    f"max_tokens={MAX_TRANSLATE_INPUT_TOKENS}"
                )
                skipped += 1
                continue

            print(
                "[번역 문단 분할] "
                f"index={row_index}, name={name}, source_language={source_lang}, "
                f"target_language={target_lang}, paragraph_tokens={segment_tokens}, "
                f"max_tokens={MAX_TRANSLATE_INPUT_TOKENS}"
            )
            sentence_chunks, sentence_skipped, _ = pack_segments_by_token_budget(
                split_sentences(segment),
                " ",
                source_lang,
                target_lang,
                processor,
                row_index=row_index,
                name=name,
                segment_kind="sentence",
            )
            split_by_sentence = True
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(sentence_chunks)
            skipped += sentence_skipped
            continue

        candidate = segment if not current else f"{current}{separator}{segment}"
        candidate_tokens = count_translation_input_tokens(candidate, source_lang, target_lang, processor)

        if candidate_tokens <= MAX_TRANSLATE_INPUT_TOKENS:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = segment

    if current:
        chunks.append(current)

    return chunks, skipped, split_by_sentence


def split_text_for_translation(
    text: str,
    source_lang: str,
    target_lang: str,
    processor,
    row_index=None,
    name: str = "",
) -> tuple[list[str], dict]:
    """source_text를 TranslateGemma 입력 한계에 맞는 chunk 목록으로 바꾼다.

    기본은 문단 단위이고, 단일 문단이 너무 길면 문장 단위로 fallback한다.
    """
    paragraphs = split_paragraphs(text)
    chunks, skipped, split_by_sentence = pack_segments_by_token_budget(
        paragraphs,
        "\n\n",
        source_lang,
        target_lang,
        processor,
        row_index=row_index,
        name=name,
        segment_kind="paragraph",
    )

    return chunks, {
        "chunk_count": len(chunks),
        "split_by_sentence": split_by_sentence,
        "skipped_overlong_segments": skipped,
    }


def generate_translation(chunk: str, source_lang: str, target_lang: str, model, processor) -> str:
    """chunk 하나를 TranslateGemma로 번역한다.

    이 함수는 이미 토큰 예산 이하로 나뉜 chunk만 받는 것을 전제로 한다.
    """
    import torch

    messages = build_translation_messages(chunk, source_lang, target_lang)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    input_len = len(inputs["input_ids"][0])

    device = get_model_device(model)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        generation = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=MAX_TRANSLATE_OUTPUT_TOKENS,
        )

    generation = generation[0][input_len:]
    return processor.decode(generation, skip_special_tokens=True).strip()


def translate_text(
    text: str,
    source_lang: str,
    target_lang: str,
    row_index=None,
    name: str = "",
) -> tuple[str, dict]:
    """긴 원문 하나를 target_lang으로 직접 번역한다.

    처리 순서:
    1. source_lang/target_lang을 언어 코드로 정규화
    2. 원문을 문단/문장 chunk로 분할
    3. 각 chunk를 원문에서 target 언어로 직접 번역
    4. chunk 순서를 유지해 다시 합침
    """
    text = str(text).strip()
    source_lang = normalize_lang_code(source_lang)
    target_lang = normalize_lang_code(target_lang)

    if not text:
        return "", {
            "chunk_count": 0,
            "split_by_sentence": False,
            "skipped_overlong_segments": 0,
        }

    if source_lang == target_lang:
        return text, {
            "chunk_count": 1,
            "split_by_sentence": False,
            "skipped_overlong_segments": 0,
        }

    if not USE_TRANSLATION:
        return "", {
            "chunk_count": 0,
            "split_by_sentence": False,
            "skipped_overlong_segments": 0,
        }

    if source_lang == "unknown":
        raise ValueError("TranslateGemma requires a known source_lang_code")

    model, processor = load_translator()
    chunks, chunk_meta = split_text_for_translation(
        text,
        source_lang,
        target_lang,
        processor,
        row_index=row_index,
        name=name,
    )

    translated_chunks = []
    failed_chunks = 0

    for chunk in chunks:
        try:
            translated = generate_translation(chunk, source_lang, target_lang, model, processor)
        except Exception as e:
            failed_chunks += 1
            print(
                "[번역 실패] "
                f"index={row_index}, name={name}, source_language={source_lang}, "
                f"target_language={target_lang}, error={e}"
            )
            translated = ""

        if translated:
            translated_chunks.append(translated)

    chunk_meta["failed_chunks"] = failed_chunks
    return "\n\n".join(translated_chunks).strip(), chunk_meta


# =========================
# 캐시
# =========================

def load_cache():
    """번역 캐시 JSON을 읽는다. 없으면 빈 dict를 반환한다."""
    if os.path.exists(TRANSLATION_CACHE):
        with open(TRANSLATION_CACHE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    """번역 캐시를 디스크에 저장한다."""
    os.makedirs(os.path.dirname(TRANSLATION_CACHE), exist_ok=True)
    with open(TRANSLATION_CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def make_source_hash(text: str) -> str:
    """긴 원문을 캐시 키에 직접 넣지 않기 위해 SHA-256 해시로 바꾼다."""
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def make_cache_key(text: str, source_lang: str, target_lang: str) -> str:
    """번역 결과를 재사용하기 위한 캐시 키를 만든다.

    모델 ID, 토큰 예산, chunk 정책이 바뀌면 예전 캐시를 재사용하지 않도록
    해당 값들을 키 재료에 포함한다.
    """
    raw = {
        "source_language": normalize_lang_code(source_lang),
        "target_language": normalize_lang_code(target_lang),
        "source_hash": make_source_hash(text),
        "model_id": TRANSLATE_MODEL_ID,
        "max_translate_input_tokens": MAX_TRANSLATE_INPUT_TOKENS,
        "chunking_policy_version": CHUNKING_POLICY_VERSION,
    }
    return hashlib.sha256(json.dumps(raw, sort_keys=True).encode("utf-8")).hexdigest()


def cached_translate(
    text: str,
    source_lang: str,
    target_lang: str,
    cache: dict,
    row_index=None,
    name: str = "",
) -> str:
    """캐시를 먼저 확인하고, 없을 때만 TranslateGemma를 호출한다."""
    source_lang = normalize_lang_code(source_lang)
    target_lang = normalize_lang_code(target_lang)
    key = make_cache_key(text, source_lang, target_lang)

    if key in cache:
        entry = cache[key]
        if isinstance(entry, dict):
            return entry.get("translated_text", "")
        return str(entry)

    translated_text, meta = translate_text(
        text,
        source_lang,
        target_lang,
        row_index=row_index,
        name=name,
    )

    cache[key] = {
        "source_language": source_lang,
        "target_language": target_lang,
        "source_hash": make_source_hash(text),
        "model_id": TRANSLATE_MODEL_ID,
        "max_translate_input_tokens": MAX_TRANSLATE_INPUT_TOKENS,
        "chunking_policy_version": CHUNKING_POLICY_VERSION,
        "source_chars": len(str(text)),
        "chunk_count": meta.get("chunk_count", 0),
        "split_by_sentence": meta.get("split_by_sentence", False),
        "skipped_overlong_segments": meta.get("skipped_overlong_segments", 0),
        "failed_chunks": meta.get("failed_chunks", 0),
        "translated_text": translated_text,
    }
    return translated_text


# =========================
# 언어 감지
# =========================

def detect_language(text: str) -> str:
    """source_text의 원문 언어를 감지한다.

    한글 비율로 먼저 한국어를 잡고, 나머지는 langdetect를 사용한다.
    """
    text = str(text).strip()

    if not text:
        return "unknown"

    hangul_count = len(re.findall(r"[가-힣]", text))
    alpha_count = len(re.findall(r"[A-Za-z]", text))

    if hangul_count >= 10 and hangul_count > alpha_count * 0.3:
        return "ko"

    try:
        from langdetect import detect
        return normalize_lang_code(detect(text))
    except Exception:
        pass

    if alpha_count > hangul_count:
        return "en"

    return "unknown"


# =========================
# 텍스트 유틸
# =========================

def split_list_field(value):
    """CSV의 쉼표 구분 tags/categories 문자열을 리스트로 바꾼다."""
    if pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    text = text.replace("[", "").replace("]", "")
    text = text.replace("'", "").replace('"', "")
    text = text.replace("|", ",")
    text = text.replace("/", ",")

    items = [x.strip() for x in text.split(",")]
    return [x for x in items if x]


def strip_markup_noise(text: str) -> str:
    """URL과 마크다운 잔여 문법처럼 검색에 방해되는 표식을 제거한다."""
    text = str(text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"`[^`]+`", " ", text)
    text = re.sub(r"#{1,6}\s*", " ", text)
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    return text


def normalize_token(text: str, prefix: str) -> str:
    """tag/category 항목을 `TAG_xxx` 또는 `CAT_xxx` 메타 토큰으로 만든다."""
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9가-힣+#]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")

    if not text:
        return ""

    return f"{prefix}_{text}"


def preprocess_english(text: str):
    """영어/라틴 문자 중심 텍스트를 공백 분리 검색 토큰으로 전처리한다."""
    text = strip_markup_noise(text).lower()
    text = re.sub(r"[^a-z0-9가-힣+#._\- ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = []

    for token in text.split():
        token = token.strip("._-")

        if not token:
            continue

        if token in EN_STOPWORDS:
            continue

        if len(token) <= 1 and not token.isdigit():
            continue

        tokens.append(token)

    return tokens


def preprocess_korean(text: str):
    """한국어 텍스트를 Okt 형태소 분석 후 검색용 품사만 남긴다."""
    text = strip_markup_noise(text)
    text = re.sub(r"[^a-zA-Z0-9가-힣+#._\- ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    pos_result = get_okt().pos(text, norm=True, stem=True)

    tokens = []

    for word, pos in pos_result:
        word = word.strip()

        if not word:
            continue

        if pos not in KEEP_KO_POS:
            continue

        if word in KO_STOPWORDS:
            continue

        if len(word) <= 1 and not word.isdigit():
            continue

        tokens.append(word)

    return tokens


def build_meta_tokens(tags, categories):
    """tags/categories를 검색 신호로 넣기 위한 메타 토큰 목록을 만든다.

    예:
    - tags의 `Kitchen Sink` -> `TAG_kitchen_sink`, `kitchen`, `sink`
    - categories의 `optimization` -> `CAT_optimization`, `optimization`
    """
    tokens = []

    for tag in tags:
        norm = normalize_token(tag, "TAG")
        if norm:
            tokens.append(norm)

        tokens += preprocess_english(tag)
        tokens += preprocess_korean(tag)

    for category in categories:
        norm = normalize_token(category, "CAT")
        if norm:
            tokens.append(norm)

        tokens += preprocess_english(category)
        tokens += preprocess_korean(category)

    seen = set()
    result = []

    for token in tokens:
        if token not in seen:
            seen.add(token)
            result.append(token)

    return result


def insert_meta(tokens, meta_tokens, interval=20, force_start=True, force_end=True):
    """언어별 토큰열의 시작, 매 interval 토큰 뒤, 끝에 meta_tokens를 삽입한다."""
    result = []

    if force_start and meta_tokens:
        result.extend(meta_tokens)

    for i, token in enumerate(tokens):
        result.append(token)

        if meta_tokens and (i + 1) % interval == 0:
            result.extend(meta_tokens)

    if force_end and meta_tokens:
        result.extend(meta_tokens)

    return result


def build_index_description(en_tokens, ko_tokens, meta_tokens):
    """최종 CSV의 description 컬럼에 들어갈 검색용 문자열을 만든다.

    출력은 별도 컬럼이 아니라 하나의 문자열이다:
    영어 토큰 + 메타 토큰 + 한국어 토큰 + 메타 토큰
    """
    result = []

    result += insert_meta(
        en_tokens,
        meta_tokens,
        interval=META_INTERVAL,
        force_start=True,
        force_end=True,
    )

    if meta_tokens:
        result += meta_tokens

    result += insert_meta(
        ko_tokens,
        meta_tokens,
        interval=META_INTERVAL,
        force_start=True,
        force_end=True,
    )

    return " ".join(result)


def build_source_description(row):
    """row의 짧은 description과 긴 body를 합쳐 번역/전처리 원문을 만든다."""
    short_desc = "" if pd.isna(row.get("description", "")) else str(row.get("description", ""))
    body = "" if pd.isna(row.get("body", "")) else str(row.get("body", ""))

    parts = []

    if short_desc.strip():
        parts.append(short_desc.strip())

    if body.strip():
        parts.append(body.strip())

    return "\n\n".join(parts).strip()


def make_en_ko_descriptions(source_text: str, lang: str, cache: dict, row_index=None, name: str = ""):
    """원문에서 영어 설명과 한국어 설명을 만든다.

    절대 영어를 중간 경유지로 쓰지 않는다.
    - 원문이 en이면 ko만 직접 번역
    - 원문이 ko이면 en만 직접 번역
    - 원문이 제3언어면 en, ko를 각각 원문에서 직접 번역
    """
    source_text = str(source_text).strip()
    lang = normalize_lang_code(lang)

    if not source_text:
        return "", ""

    if lang == "en":
        english_text = source_text
        try:
            korean_text = cached_translate(
                source_text,
                "en",
                "ko",
                cache,
                row_index=row_index,
                name=name,
            )
        except Exception as e:
            print(
                "[번역 실패] "
                f"index={row_index}, name={name}, source_language=en, "
                f"target_language=ko, error={e}"
            )
            korean_text = ""

    elif lang == "ko":
        korean_text = source_text
        try:
            english_text = cached_translate(
                source_text,
                "ko",
                "en",
                cache,
                row_index=row_index,
                name=name,
            )
        except Exception as e:
            print(
                "[번역 실패] "
                f"index={row_index}, name={name}, source_language=ko, "
                f"target_language=en, error={e}"
            )
            english_text = source_text

    elif lang == "unknown":
        print(f"[언어 감지 실패] index={row_index}, name={name}")
        english_text = source_text
        korean_text = ""

    else:
        try:
            english_text = cached_translate(
                source_text,
                lang,
                "en",
                cache,
                row_index=row_index,
                name=name,
            )
        except Exception as e:
            print(
                "[번역 실패] "
                f"index={row_index}, name={name}, source_language={lang}, "
                f"target_language=en, error={e}"
            )
            english_text = source_text

        try:
            korean_text = cached_translate(
                source_text,
                lang,
                "ko",
                cache,
                row_index=row_index,
                name=name,
            )
        except Exception as e:
            print(
                "[번역 실패] "
                f"index={row_index}, name={name}, source_language={lang}, "
                f"target_language=ko, error={e}"
            )
            korean_text = ""

    return english_text, korean_text


# =========================
# 메인
# =========================

def main():
    """전처리 작업의 진입점이다.

    이 함수가 하는 일:
    1. `INPUT_CSV`를 읽고 필수 컬럼을 확인한다.
    2. 각 row의 description/body를 영어+한국어 검색 토큰으로 변환한다.
    3. 원본 DataFrame의 description 컬럼만 교체한다.
    4. `OUTPUT_CSV`로 저장한다.
    """
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"입력 CSV 없음: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"필수 컬럼 없음: {missing_cols}. 현재 컬럼: {list(df.columns)}")

    cache = load_cache()
    processed_descriptions = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
        name = str(row.get("name", "")).strip()

        source_text = build_source_description(row)
        if not source_text:
            processed_descriptions.append("")
            continue

        lang = detect_language(source_text)
        en_text, ko_text = make_en_ko_descriptions(
            source_text,
            lang,
            cache,
            row_index=idx,
            name=name,
        )

        tags = split_list_field(row.get("tags", ""))
        categories = split_list_field(row.get("categories", ""))

        meta_tokens = build_meta_tokens(tags, categories)
        en_tokens = preprocess_english(en_text)
        ko_tokens = preprocess_korean(ko_text)

        final_description = build_index_description(
            en_tokens=en_tokens,
            ko_tokens=ko_tokens,
            meta_tokens=meta_tokens,
        )

        processed_descriptions.append(final_description)

        if idx > 0 and idx % 20 == 0:
            save_cache(cache)

    save_cache(cache)

    out_df = df.copy()
    out_df["description"] = processed_descriptions

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"완료: {OUTPUT_CSV}")
    print(out_df[["name", "description", "tags", "categories"]].head(3).to_string())


if __name__ == "__main__":
    main()
