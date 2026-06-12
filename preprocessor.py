"""Modrinth modpack CSV preprocessor.

이 파일이 하는 일:
- `INPUT_CSV`에서 크롤러가 만든 원본 CSV를 읽는다.
- 각 row의 `description + body`를 원문 설명으로 만들고 HTML/Markdown 노이즈를 정리한다.
- 원문에서 영어 설명과 한국어 설명을 만든다.
- 영어/한국어 설명을 검색용 토큰으로 바꾸고 불용어를 제거한다.
- tags/categories를 메타 토큰으로 만들어 토큰열 시작, 매 20토큰, 끝, 영한 경계에 삽입한다.
- 출력 CSV에는 원본 컬럼 순서를 유지하고 `description` 컬럼 값만 전처리 결과로 교체한다.

이 파일에 넣으면 안 되는 일:
- Hugging Face 모델 로드/양자화/번역 generation 구현은 `translategemma_translator.py`에 둔다.
- Word2Vec/TF-IDF 학습은 `generate_model.py`에서 한다.
- 추천/필터/랭킹은 추천 모듈에서 한다.

함수로 남긴 기준:
- HTML/Markdown 정리, 토큰화, 메타 토큰 삽입처럼 규칙이 길고 따로 검증할 가치가 있는 단위는 함수로 둔다.
- row 하나를 처리하는 큰 흐름은 `main()` 안에 풀어 두어 위에서 아래로 읽히게 한다.
"""

from __future__ import annotations

import html
import re
import time
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd

from translategemma_translator import (
    TranslateGemmaConfig,
    TranslateGemmaTranslator,
    load_translation_cache,
    save_translation_cache,
)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        """tqdm이 없을 때 진행바 없이 그대로 반복하기 위한 fallback이다."""
        return iterable


# =========================
# 경로와 번역 설정
# =========================

# 크롤러가 만든 원본 CSV 경로를 넣는다.
INPUT_CSV = "./datasets/modrinth_dataset.csv"
# 이 스크립트가 저장할 전처리 CSV 경로다. 기존 description 컬럼 값만 교체된다.
OUTPUT_CSV = "./datasets/modrinth_dataset_preprocessed.csv"
# TranslateGemma 결과 캐시 파일이다. 같은 설명을 매번 다시 번역하지 않기 위해 쓴다.
TRANSLATION_CACHE = "./datasets/translation_cache.json"

# Hugging Face에서 받을 TranslateGemma 모델 ID를 넣는다.
TRANSLATE_MODEL_ID = "google/translategemma-4b-it"
# None이면 ./models/<safe_model_id>를 쓴다. 이미 받은 모델 폴더를 직접 지정할 때만 넣는다.
TRANSLATE_MODEL_DIR = None
# 번역 없이 CSV 정리/토큰화 흐름만 확인하고 싶으면 False로 바꾼다.
USE_TRANSLATION = True

# 지원값: none, 8bit, 4bit. 양자화는 모델 다운로드가 아니라 로드 시점 설정이다.
TRANSLATE_QUANTIZATION = "4bit"
MODEL_DTYPE = "bfloat16"
TRANSLATE_DEVICE_MAP = None

# bitsandbytes 4bit/8bit 세부 설정. 잘 모르면 기본값을 유지한다.
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_COMPUTE_DTYPE = "bfloat16"
BNB_4BIT_USE_DOUBLE_QUANT = True
BNB_8BIT_THRESHOLD = 6.0

# 번역 chunk 예산.
# 입력 토큰은 원문+chat template이 모델에 들어가는 길이이고,
# 출력 토큰은 번역문으로 새로 생성되는 길이다. 출력이 부족하면 로그가 뜬다.
MAX_TRANSLATE_INPUT_TOKENS = 512
# 일반 chunk 예산을 넘는 단일 문장도 이 절대 한도 이하이면 번역을 시도한다.
HARD_MAX_TRANSLATE_INPUT_TOKENS = 1800
MAX_TRANSLATE_OUTPUT_TOKENS = 512
MIN_TRANSLATE_OUTPUT_TOKENS = 64
OUTPUT_TOKEN_RATIO = 1.5

# 번역이 비정상적으로 길어지거나 실패할 때 실제 입력/출력을 출력하고 파일로 남긴다.
TRANSLATE_DEBUG_ON_FAILURE = True
TRANSLATE_DEBUG_TO_CONSOLE = True
TRANSLATE_DEBUG_DIR = "./logs/translation_debug"
TRANSLATE_DEBUG_MAX_CHARS = 4000
TRANSLATE_LOG_PROGRESS = True
TRANSLATE_LOG_PREVIEW_CHARS = 120

# tags/categories 메타 토큰을 언어별 토큰열 몇 개마다 다시 끼워 넣을지 정한다.
META_INTERVAL = 20
# 캐시 저장/row 처리 시간 로그 설정.
CACHE_SAVE_EVERY = 20
PRINT_ROW_TIMING = True


# =========================
# 입력 CSV 컬럼
# =========================

# 입력 CSV에 반드시 있어야 하는 컬럼이다.
# 이 스크립트는 아래 기존 컬럼을 보존하고, description 값만 바꾼다.
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


# =========================
# 불용어
# =========================

# 한국어 번역 결과에서 검색 의미가 약한 일반 표현을 제거한다.
# 장르/의도 단어(퀘스트, 탐험, 자동화, 최적화 등)는 여기에 넣지 않는다.
KO_STOPWORDS = {
    "것", "수", "등", "때", "곳", "거", "점", "중", "전", "후",
    "내", "외", "위", "뒤", "더", "또", "및", "좀", "매우",
    "그리고", "그러나", "하지만", "또한", "또는", "혹은",
    "하다", "되다", "있다", "없다", "같다", "아니다",
    "위해", "위한", "대해", "대한", "통한", "통해", "관련",
    "가능", "필요", "경우", "부분", "내용", "페이지",
    "마인크래프트", "마크", "모드팩", "팩", "모드",
    "공식", "버전", "포함", "추가", "제공", "사용", "플레이",
    "경험", "패키지", "구성", "기반", "중심",
    "당신", "여러", "모든", "많은", "새로운", "간단하다",
    "패브릭", "포지", "네오포지", "퀼트",
}

# 영어 설명에서 너무 자주 반복되어 검색 구분력이 낮은 단어를 제거한다.
# `create`, `vanilla`, `quest`, `performance`, `optimization`, `server`,
# `world`, `adventure`처럼 모드팩 검색 의도가 되는 단어는 보존한다.
EN_STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "where", "when", "why", "how",
    "a", "an", "the", "and", "or", "but", "if", "then", "than",
    "because", "while", "until", "again", "about", "above", "below",
    "between", "through", "during", "before", "after", "over", "under",
    "out", "up", "down", "in", "on", "off", "for", "with", "by", "from",
    "to", "of", "as", "at", "into", "within", "without",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing",
    "will", "can", "could", "should", "would", "may", "might", "must", "shall",
    "this", "that", "these", "those", "there", "here",
    "no", "nor", "not", "only", "own", "same", "such", "so", "too", "very",
    "just", "now", "also", "some", "any", "both", "each", "few", "most",
    "other", "another", "more", "less", "many", "one", "two", "three", "even",
    "use", "used", "using", "make", "makes", "made", "get", "gets", "got",
    "want", "wants", "like",
    "minecraft", "modpack", "modpacks", "mod", "mods",
    "pack", "packs", "package", "packages",
    "include", "includes", "included", "including",
    "feature", "features", "content", "experience",
    "official", "version", "versions", "new", "all", "everything",
    "support", "supports", "supported", "supporting",
    "need", "needs", "built", "designed", "based", "focused",
    "simple", "proper", "true", "best", "popular", "better",
    "play", "playing", "game", "mc", "modrinth", "discord",
    "config", "configuration", "menu", "add", "adds", "added", "adding",
    "fabric", "forge", "neoforge", "quilt",
}


# =========================
# HTML/Markdown 정리
# =========================

class HtmlToText(HTMLParser):
    """HTML 조각을 텍스트로 바꾸되 문단/목록 경계를 최대한 보존한다."""

    BLOCK_TAGS = {
        "address", "article", "aside", "blockquote", "br", "dd", "details",
        "div", "dl", "dt", "figcaption", "figure", "footer", "h1", "h2",
        "h3", "h4", "h5", "h6", "header", "hr", "li", "main", "ol", "p",
        "pre", "section", "summary", "table", "tbody", "td", "tfoot", "th",
        "thead", "tr", "ul",
    }
    SKIP_TAGS = {"script", "style", "iframe", "svg", "video", "audio", "picture"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag, _attrs):
        tag = tag.lower()
        if tag in self.SKIP_TAGS:
            self.skip_depth += 1
            return
        if tag == "img":
            return
        if tag == "li":
            self.parts.append("\n- ")
        elif tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in self.SKIP_TAGS and self.skip_depth:
            self.skip_depth -= 1
            return
        if not self.skip_depth and tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip_depth:
            self.parts.append(data)

    def get_text(self) -> str:
        return "".join(self.parts)


def normalize_spacing(text: str) -> str:
    """문단 경계는 살리고 줄 내부의 과도한 공백만 정리한다."""
    lines = []
    for line in str(text).replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = re.sub(r"[ \t\f\v]+", " ", line).strip()
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_text(text: str) -> str:
    """번역 전에 Modrinth description/body의 HTML/Markdown/링크 노이즈를 제거한다."""
    text = html.unescape("" if pd.isna(text) else str(text))

    # 코드/이미지/링크를 먼저 정리한다. URL은 검색 의미보다 노이즈가 큰 경우가 많다.
    text = re.sub(r"```.*?```|~~~.*?~~~", "\n", text, flags=re.DOTALL)
    text = re.sub(r"\[!\[[^\]]*](?:\([^)]+\)|\[[^\]]+])]\([^)]+\)", "\n", text)
    text = re.sub(r"!\[[^\]]*]\([^)]+\)", "\n", text)
    text = re.sub(r"!\[[^\]]*]\[[^\]]+]", "\n", text)
    text = re.sub(r"\[([^\]]+)]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)]\[[^\]]+]", r"\1", text)
    text = re.sub(r"(?m)^\s*\[[^\]]+]:\s*\S+.*$", "\n", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Markdown 장식 문법은 제거하되 제목/목록의 텍스트 자체는 보존한다.
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "\n", text)
    text = re.sub(r"(?m)^\s{0,3}>\s?", "", text)
    text = re.sub(r"(?m)^\s*[-*_]{3,}\s*$", "\n", text)
    text = re.sub(r"(?m)^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$", "\n", text)
    text = re.sub(r"(?m)^\s*[-*+]\s+", "- ", text)
    text = re.sub(r"(?m)^\s*\d+[.)]\s+", "- ", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"[*_]{1,3}([^*_]+)[*_]{1,3}", r"\1", text)
    text = text.replace("|", " ")

    if re.search(r"</?[A-Za-z][^>]*>", text):
        parser = HtmlToText()
        parser.feed(text)
        parser.close()
        text = parser.get_text()

    return normalize_spacing(text)


# =========================
# 언어 감지와 토큰화
# =========================

TOKEN_RE = re.compile(r"[A-Za-z0-9가-힣][A-Za-z0-9가-힣+#._-]*")


def detect_language(text: str) -> str:
    """source_text의 주 언어를 간단히 추정한다.

    Modrinth 데이터는 대부분 영어이므로, 한글/일본어/중국어 신호가 약하면 영어로 본다.
    """
    text = str(text)
    hangul = len(re.findall(r"[가-힣]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    kana = len(re.findall(r"[\u3040-\u30ff]", text))
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))

    if hangul >= 10 and hangul > latin * 0.3:
        return "ko"
    if kana >= 10 and kana > latin * 0.3:
        return "ja"
    if cjk >= 10 and cjk > latin * 0.3:
        return "zh"
    if latin > 0:
        return "en"
    return "unknown"


def strip_token_noise(text: str) -> str:
    """번역 이후 남아 있을 수 있는 URL/문장부호 노이즈를 토큰화 전에 한 번 더 정리한다."""
    text = re.sub(r"https?://\S+|www\.\S+", " ", str(text))
    text = re.sub(r"[`*_~<>{}\[\]()/\\|=]", " ", text)
    return text.lower()


def preprocess_english(text: str) -> list[str]:
    """영어 설명을 공백 분리 검색 토큰으로 만든다.

    여기에 들어가는 값:
    - TranslateGemma 번역 전/후의 영어 설명 문자열

    반환값:
    - TF-IDF/Word2Vec 학습에 그대로 split해서 쓸 토큰 리스트
    """
    tokens: list[str] = []
    for raw in TOKEN_RE.findall(strip_token_noise(text)):
        token = raw.strip("._-")
        if not token:
            continue
        if token in EN_STOPWORDS:
            continue
        if len(token) <= 1 and not token.isdigit():
            continue
        tokens.append(token)
    return tokens


def preprocess_korean(text: str) -> list[str]:
    """한국어 설명을 공백 분리 검색 토큰으로 만든다.

    현재는 추가 Java/konlpy 의존성을 피하기 위해 정규식 토큰화와 불용어 제거만 한다.
    이후 `generate_model.py`에서 형태소 분석을 붙일 수 있지만, CSV 전처리기는
    여기서 이미 영어+한국어 검색 신호를 한 컬럼에 만들어 둔다.
    """
    tokens: list[str] = []
    for raw in TOKEN_RE.findall(strip_token_noise(text)):
        token = raw.strip("._-")
        if not token:
            continue
        if re.search(r"[가-힣]", token) and token in KO_STOPWORDS:
            continue
        if re.fullmatch(r"[a-z0-9+#._-]+", token) and token in EN_STOPWORDS:
            continue
        if len(token) <= 1 and not token.isdigit():
            continue
        tokens.append(token)
    return tokens


# =========================
# tags/categories 메타 토큰
# =========================

def normalize_meta_value(value: str) -> str:
    """메타 토큰 값에 들어갈 문자열을 소문자/언더스코어 형태로 맞춘다."""
    value = str(value).strip().lower()
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^a-z0-9가-힣+#._-]+", "", value)
    return value.strip("._-")


def meta_tokens(row) -> list[str]:
    """tags/categories를 검색 신호로 넣기 위한 메타 토큰 목록을 만든다."""
    tokens = []

    raw_tags = [] if pd.isna(row.get("tags", "")) else str(row.get("tags", "")).split(",")
    raw_categories = [] if pd.isna(row.get("categories", "")) else str(row.get("categories", "")).split(",")

    for tag in raw_tags:
        value = normalize_meta_value(tag)
        if value:
            tokens.append(f"tag:{value}")
    for category in raw_categories:
        value = normalize_meta_value(category)
        if value:
            tokens.append(f"category:{value}")
    return list(dict.fromkeys(tokens))


def insert_meta(tokens: list[str], meta: list[str]) -> list[str]:
    """언어별 토큰열의 시작, 매 META_INTERVAL 토큰 뒤, 끝에 meta 토큰을 삽입한다."""
    if not meta:
        return tokens

    output = meta[:]
    for index, token in enumerate(tokens, 1):
        output.append(token)
        if index % META_INTERVAL == 0:
            output.extend(meta)
    output.extend(meta)
    return output


# =========================
# 전체 처리 흐름
# =========================


def main():
    # 1. 원본 CSV가 있는지 확인하고 읽는다.
    if not Path(INPUT_CSV).exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    # 2. 크롤러가 만들어야 하는 필수 컬럼이 빠졌는지 확인한다.
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # 3. 번역기 설정을 만든다. 실제 모델 로드/양자화/다운로드 코드는 별도 파일에 있다.
    translator = TranslateGemmaTranslator(TranslateGemmaConfig(
        model_id=TRANSLATE_MODEL_ID,
        model_dir=TRANSLATE_MODEL_DIR,
        cache_path=TRANSLATION_CACHE,
        use_translation=USE_TRANSLATION,
        quantization=TRANSLATE_QUANTIZATION,
        model_dtype=MODEL_DTYPE,
        device_map=TRANSLATE_DEVICE_MAP,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=BNB_4BIT_COMPUTE_DTYPE,
        bnb_4bit_use_double_quant=BNB_4BIT_USE_DOUBLE_QUANT,
        bnb_8bit_threshold=BNB_8BIT_THRESHOLD,
        max_input_tokens=MAX_TRANSLATE_INPUT_TOKENS,
        hard_max_input_tokens=HARD_MAX_TRANSLATE_INPUT_TOKENS,
        max_output_tokens=MAX_TRANSLATE_OUTPUT_TOKENS,
        min_output_tokens=MIN_TRANSLATE_OUTPUT_TOKENS,
        output_token_ratio=OUTPUT_TOKEN_RATIO,
        debug_on_failure=TRANSLATE_DEBUG_ON_FAILURE,
        debug_to_console=TRANSLATE_DEBUG_TO_CONSOLE,
        debug_dir=TRANSLATE_DEBUG_DIR,
        debug_max_chars=TRANSLATE_DEBUG_MAX_CHARS,
        log_progress=TRANSLATE_LOG_PROGRESS,
        log_preview_chars=TRANSLATE_LOG_PREVIEW_CHARS,
    ))
    if USE_TRANSLATION:
        translator.load()

    # 4. 이미 번역한 원문은 캐시에서 재사용한다.
    cache = load_translation_cache(TRANSLATION_CACHE)
    descriptions: list[str] = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
        start = time.perf_counter()

        # 5. 짧은 description과 긴 body를 각각 정리한 뒤 하나의 원문으로 합친다.
        #    \n\n은 번역기가 둘을 다른 문단으로 볼 수 있게 하는 경계다.
        source_parts = [
            clean_text(row.get("description", "")),
            clean_text(row.get("body", "")),
        ]
        source_text = normalize_spacing("\n\n".join(part for part in source_parts if part))

        if not source_text:
            descriptions.append("")
        else:
            # 6. 원문 언어를 판정한다. 영어/한국어가 아닌 경우에도 2단 번역은 하지 않는다.
            lang = detect_language(source_text)
            row_label = f"row={idx} name={str(row.get('name', '')).strip()[:80]}"

            # 7. 영어 설명과 한국어 설명을 만든다.
            #    원문이 이미 목표 언어면 그대로 쓰고, 필요한 쪽만 원문에서 직접 번역한다.
            if not USE_TRANSLATION:
                en_text = source_text
                ko_text = ""
            elif lang == "ko":
                ko_text = source_text
                en_text = translator.translate_text(source_text, "ko", "en", cache, label=row_label) or source_text
            elif lang == "en":
                en_text = source_text
                ko_text = translator.translate_text(source_text, "en", "ko", cache, label=row_label)
            elif lang in {"ja", "zh"}:
                en_text = translator.translate_text(source_text, lang, "en", cache, label=row_label) or source_text
                ko_text = translator.translate_text(source_text, lang, "ko", cache, label=row_label)
            else:
                print("[language unknown] source language unknown; keeping source as english_text and skipping korean translation")
                en_text = source_text
                ko_text = ""

            # 8. tags/categories를 검색용 메타 토큰으로 만든다.
            meta = meta_tokens(row)

            # 9. 영어/한국어 텍스트를 각각 검색용 토큰으로 바꾸고, 각 언어 토큰열에 메타 토큰을 삽입한다.
            en_tokens = insert_meta(preprocess_english(en_text), meta)
            ko_tokens = insert_meta(preprocess_korean(ko_text), meta)

            # 10. 최종 출력은 별도 영어/한국어 컬럼이 아니라 기존 description 한 칸이다.
            #     영한 경계에도 meta를 한 번 더 넣어 어느 쪽 텍스트로 검색해도 태그 신호가 남게 한다.
            descriptions.append(" ".join(en_tokens + (meta if meta else []) + ko_tokens).strip())

        if PRINT_ROW_TIMING:
            print(f"[row] index={idx} elapsed={time.perf_counter() - start:.1f}s")
        if idx and idx % CACHE_SAVE_EVERY == 0:
            save_translation_cache(cache, TRANSLATION_CACHE)

    # 11. 마지막 캐시를 저장하고, 원본 컬럼 순서를 유지한 채 description 값만 교체한다.
    save_translation_cache(cache, TRANSLATION_CACHE)

    out_df = df.copy()
    out_df["description"] = descriptions
    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"done: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
