"""Modrinth modpack CSV preprocessor.

이 파일이 하는 일:
- `INPUT_CSV`에서 크롤러가 만든 원본 CSV를 읽는다.
- 각 row의 `description + body`를 원문 설명으로 만들고 HTML/Markdown 노이즈를 정리한다.
- 원문이 영어가 아니면 영어로 번역하고, 영어 검색용 토큰으로 전처리한다.
- tags/categories를 메타 토큰으로 만들어 토큰열 시작, 매 20토큰, 끝에 삽입한다.
- 출력 CSV에는 원본 컬럼 순서를 유지하고 `description`을 교체한 뒤 `complete` 컬럼을 추가한다.

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
from langdetect import DetectorFactory, LangDetectException, detect_langs
from lemminflect import getLemma
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as SKLEARN_STOP_WORDS

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
TRANSLATE_QUANTIZATION = "8bit"
MODEL_DTYPE = "float16"
TRANSLATE_DEVICE_MAP = None

# bitsandbytes 4bit/8bit 세부 설정. 잘 모르면 기본값을 유지한다.
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_COMPUTE_DTYPE = "bfloat16"
BNB_4BIT_USE_DOUBLE_QUANT = True
BNB_8BIT_THRESHOLD = 6.0

# 번역 chunk 예산.
# 입력 토큰은 원문+chat template이 모델에 들어가는 길이이고,
# 출력 토큰은 번역문으로 새로 생성되는 길이다. 출력이 부족하면 로그가 뜬다.
MAX_TRANSLATE_INPUT_TOKENS = 1024
# 일반 chunk 예산을 넘는 단일 문장도 이 절대 한도 이하이면 번역을 시도한다.
HARD_MAX_TRANSLATE_INPUT_TOKENS = 1800
MAX_TRANSLATE_OUTPUT_TOKENS = 2000
MIN_TRANSLATE_OUTPUT_TOKENS = 1
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

DetectorFactory.seed = 0
LANG_ALIASES = {
    "en": "en",
    "ko": "ko",
    "ja": "ja",
    "zh-cn": "zh",
    "zh-tw": "zh",
    "zh": "zh",
}
LEMMA_UPOS_ORDER = ("NOUN", "VERB", "ADJ", "ADV")
TOKEN_ALIASES = {
    "optimise": "optimize",
    "optimisation": "optimization",
}
SCRIPT_LANGUAGE_RE = (
    ("ko", re.compile(r"[가-힣]")),
    ("ja", re.compile(r"[\u3040-\u30ff]")),
    ("zh", re.compile(r"[\u4e00-\u9fff]")),
)
MIN_SCRIPT_CHARS = 6
MIN_SCRIPT_TO_LATIN_RATIO = 0.12

# 마인크래프트 생태계 고유명사와 검색 의도 단어는 불용어보다 우선 보존한다.
PROTECTED_TERMS = {
    "adventure", "ae2", "animation", "apocalypse", "api", "applied", "ars",
    "automation", "backpack", "biome", "block", "boss", "botania", "bukkit",
    "c2me", "cave", "challenging", "chat", "client", "clientside",
    "cobblemon", "combat", "create", "cursed", "datapack", "decoration",
    "dimension", "dungeon", "economy", "embeddium", "end", "energistics",
    "engineering", "entity", "equipment", "exploration", "fabric", "factory",
    "fantasy", "ferritecore", "food", "forge", "fps", "ftb", "gui",
    "hardcore", "horror", "hud", "immersive", "indium", "industrial",
    "inventory", "iris", "item", "kitchen", "krypton", "kubejs", "library",
    "lightweight", "lithium", "magic", "management", "map", "medieval",
    "mekanism", "minigame", "minimap", "mmorpg", "mob", "modernfix",
    "modloader", "multiplayer", "neoforge", "nether", "noisium", "nvidium",
    "oculus", "ore", "origin", "origins", "overworld", "paper",
    "performance", "phosphor", "pokemon", "progression", "purpur", "pve",
    "pvp", "qol", "quest", "quilt", "refined", "resource", "resourcepack",
    "rpg", "server", "serverside", "shader", "singleplayer", "sink",
    "skyblock", "smp", "social", "sodium", "spigot", "starlight", "storage",
    "structure", "survival", "tech", "technology", "terrain", "texture",
    "thermal", "transportation", "utility", "vanilla", "vanilla+", "visual",
    "voice", "vulkan", "vulkanmod", "waystone", "world", "worldgen", "yung",
    "zombie",
}

# 영어 설명에서 너무 자주 반복되어 추천 구분력이 낮은 기능어와 소개 문구를 제거한다.
# PROTECTED_TERMS에 있는 단어는 여기에 들어 있어도 최종적으로 보존된다.
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
    "want", "wants", "like", "enjoy", "find", "join",
    "minecraft", "modpack", "modpacks", "mod", "mods",
    "pack", "packs", "package", "packages",
    "include", "includes", "included", "including",
    "feature", "features", "content", "experience",
    "official", "version", "versions", "new", "all", "everything",
    "support", "supports", "supported", "supporting",
    "need", "needs", "built", "designed", "based", "focused",
    "simple", "proper", "true", "best", "popular", "better",
    "play", "playing", "game", "mc", "modrinth", "discord",
    "player", "players", "thing", "things", "stuff",
    "config", "configuration", "menu", "add", "adds", "added", "adding",
    "summary", "details", "description", "information", "note", "important",
    "please", "click", "link", "links", "list", "title", "section",
    "recommend", "recommends", "recommended", "recommendation",
    "install", "installs", "installed", "installing", "installation",
    "available", "default", "optional", "required", "compatible",
    "launcher", "profile", "settings", "issue", "issues", "report",
    "read", "work", "works", "working", "run", "runs", "running",
    "able", "allow", "allows", "try", "today", "thanks", "welcome",
    "smooth", "smoother", "improved", "improvement", "improvements",
    "quality", "life", "ram", "gb", "mb", "memory", "allocate", "allocated",
    "download", "downloads", "http", "https", "www", "com", "src", "href",
    "png", "jpg", "jpeg", "gif", "image", "images", "logo", "readme",
    "curseforge", "github", "wiki",
}
EN_STOPWORDS = (EN_STOPWORDS | set(SKLEARN_STOP_WORDS)) - PROTECTED_TERMS


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
    SKIP_TAGS = {
        "script", "style", "iframe", "svg", "video", "audio", "picture",
        "pre", "code", "kbd", "samp",
    }

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
    text = re.sub(r"<!--.*?-->", "\n", text, flags=re.DOTALL)
    text = re.sub(r"```.*?```|~~~.*?~~~", "\n", text, flags=re.DOTALL)
    text = re.sub(r"(?m)^\s{4,}\S.*$", "\n", text)
    text = re.sub(r"<(?:script|style|iframe|svg|video|audio|picture|pre|code|kbd|samp)\b[^>]*>.*?</(?:script|style|iframe|svg|video|audio|picture|pre|code|kbd|samp)>", "\n", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"\[!\[[^\]]*](?:\([^)]+\)|\[[^\]]+])]\([^)]+\)", "\n", text)
    text = re.sub(r"!\[[^\]]*]\([^)]+\)", "\n", text)
    text = re.sub(r"!\[[^\]]*]\[[^\]]+]", "\n", text)
    text = re.sub(r"\[([^\]]+)]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)]\[[^\]]+]", r"\1", text)
    text = re.sub(r"(?m)^\s*\[[^\]]+]:\s*\S+.*$", "\n", text)
    text = re.sub(r"<?(?:https?://[^\s<>)]+|www\.[^\s<>)]+|mailto:[^\s<>)]+)>?", " ", text)

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

TOKEN_RE = re.compile(r"[A-Za-z0-9+#][A-Za-z0-9+#._-]*")
VERSION_RE = re.compile(r"^\d+(?:\.\d+){1,3}$")
PHRASE_REPLACEMENTS = (
    (re.compile(r"\bquality\s+of\s+life\b", re.IGNORECASE), "qol"),
    (re.compile(r"\bclient\s+side\b", re.IGNORECASE), "clientside"),
    (re.compile(r"\bserver\s+side\b", re.IGNORECASE), "serverside"),
    (re.compile(r"\bresource\s+pack\b", re.IGNORECASE), "resourcepack"),
    (re.compile(r"\bdata\s+pack\b", re.IGNORECASE), "datapack"),
    (re.compile(r"\bmod\s+loader\b", re.IGNORECASE), "modloader"),
    (re.compile(r"\bworld\s+gen(?:eration)?\b", re.IGNORECASE), "worldgen"),
    (re.compile(r"\bkitchen\s+sink\b", re.IGNORECASE), "kitchen sink"),
    (re.compile(r"\bvanilla\s+plus\b", re.IGNORECASE), "vanilla+"),
)


def detect_language(text: str) -> str:
    """langdetect로 source_text의 주 언어를 추정한다."""
    sample = normalize_spacing(text)[:4000]
    if not sample:
        return "unknown"
    latin_count = len(re.findall(r"[A-Za-z]", sample))
    for lang, pattern in SCRIPT_LANGUAGE_RE:
        script_count = len(pattern.findall(sample))
        if script_count >= MIN_SCRIPT_CHARS and script_count >= latin_count * MIN_SCRIPT_TO_LATIN_RATIO:
            return lang
    try:
        candidates = detect_langs(sample)
    except LangDetectException:
        return "unknown"
    if not candidates:
        return "unknown"
    best = candidates[0]
    if best.prob < 0.60:
        return "unknown"
    return LANG_ALIASES.get(best.lang.lower(), best.lang.split("-", 1)[0].lower())


def strip_token_noise(text: str) -> str:
    """번역 이후 남아 있을 수 있는 URL/문장부호/마크다운 노이즈를 토큰화 전에 한 번 더 정리한다."""
    text = html.unescape(str(text))
    for pattern, replacement in PHRASE_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"<?(?:https?://[^\s<>)]+|www\.[^\s<>)]+|mailto:[^\s<>)]+)>?", " ", text)
    text = re.sub(r"[`*_~<>{}\[\]()/\\|=:$;,!?\"']", " ", text)
    return text.lower()


def split_english_terms(raw: str) -> list[str]:
    """하이픈/언더스코어 복합어는 검색 가능한 작은 단어로 나눈다."""
    token = raw.strip("._-").lower()
    if not token:
        return []
    if VERSION_RE.fullmatch(token):
        return [token]
    return [part for part in re.split(r"[._-]+", token) if part]


def lemmatize_english_token(token: str) -> str:
    """LemmInflect로 영어 토큰을 자동 원형화한다."""
    if token in PROTECTED_TERMS or VERSION_RE.fullmatch(token) or any(ch.isdigit() for ch in token):
        return token
    for upos in LEMMA_UPOS_ORDER:
        lemmas = getLemma(token, upos=upos)
        if lemmas:
            lemma = lemmas[0].lower()
            return TOKEN_ALIASES.get(lemma, lemma)
    return TOKEN_ALIASES.get(token, token)


def preprocess_english(text: str) -> list[str]:
    """영어 설명을 TF-IDF/Word2Vec 학습용 공백 분리 토큰으로 만든다."""
    tokens: list[str] = []
    for raw in TOKEN_RE.findall(strip_token_noise(text)):
        for part in split_english_terms(raw):
            token = lemmatize_english_token(part)
            if token in PROTECTED_TERMS:
                tokens.append(token)
                continue
            if token in EN_STOPWORDS:
                continue
            if len(token) <= 1 and not token.isdigit():
                continue
            if token.isdigit() and len(token) > 4:
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


def build_source_text(row) -> str:
    """description과 body를 정리해 번역/전처리에 넣을 원문을 만든다."""
    source_parts = [
        clean_text(row.get("description", "")),
        clean_text(row.get("body", "")),
    ]
    return normalize_spacing("\n\n".join(part for part in source_parts if part))


def csv_text(value) -> str:
    """CSV의 빈 값을 문자열 'nan'으로 바꾸지 않고 읽는다."""
    return "" if pd.isna(value) else str(value)


def row_key(row) -> str:
    """재시도 판단에 쓸 안정적인 row 키를 만든다."""
    return csv_text(row.get("slug", "")) or csv_text(row.get("url", ""))


def is_complete_value(value) -> bool:
    """complete 컬럼에서 1/true 계열만 완료로 인정한다."""
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def output_columns(df: pd.DataFrame) -> list[str]:
    """원본 컬럼 뒤에 complete 컬럼만 추가한다."""
    return list(df.columns) + (["complete"] if "complete" not in df.columns else [])


def normalize_existing_output(output_csv: str, columns: list[str]) -> dict[str, bool]:
    """기존 출력 CSV를 읽어 slug별 최신 complete 상태를 만든다."""
    path = Path(output_csv)
    if not path.exists() or path.stat().st_size == 0:
        return {}

    out_df = pd.read_csv(path)
    if "complete" not in out_df.columns:
        out_df["complete"] = 0
        out_df.to_csv(path, index=False, encoding="utf-8-sig")

    status: dict[str, bool] = {}
    for _idx, row in out_df.iterrows():
        key = row_key(row)
        if key:
            status[key] = is_complete_value(row.get("complete", 0))
    return status


def append_output_row(row, processed_description: str, complete: int, columns: list[str]) -> None:
    """row 처리 결과를 최종 출력 CSV에 즉시 append한다."""
    path = Path(OUTPUT_CSV)
    path.parent.mkdir(parents=True, exist_ok=True)

    record = row.to_dict()
    record["description"] = processed_description
    record["complete"] = int(complete)

    write_header = not path.exists() or path.stat().st_size == 0
    pd.DataFrame([record], columns=columns).to_csv(
        path,
        mode="a",
        header=write_header,
        index=False,
        encoding="utf-8-sig",
    )


def build_processed_description(row, english_text: str) -> str:
    """영어 텍스트와 메타 토큰으로 최종 description 문자열을 만든다."""
    meta = meta_tokens(row)
    en_tokens = insert_meta(preprocess_english(english_text), meta)
    return " ".join(en_tokens).strip()


def translate_source_text(translator, source_text: str, lang: str, cache: dict[str, str], row_label: str) -> str:
    """영어가 아닌 source_text만 영어로 번역한다."""
    if not USE_TRANSLATION:
        return source_text
    if lang in {"en", "unknown"}:
        return source_text
    return translator.translate_text(
        source_text,
        lang,
        "en",
        cache,
        label=row_label,
        raise_on_failure=True,
    ) or source_text


def make_translator() -> TranslateGemmaTranslator:
    """전처리와 검색이 같은 TranslateGemma 설정을 쓰게 번역기를 만든다."""
    return TranslateGemmaTranslator(TranslateGemmaConfig(
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


# =========================
# 전체 처리 흐름
# =========================


def main():
    # 1. 원본 CSV가 있는지 확인하고 읽는다.
    if not Path(INPUT_CSV).exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    columns = output_columns(df)
    complete_by_key = normalize_existing_output(OUTPUT_CSV, columns)

    # 2. 크롤러가 만들어야 하는 필수 컬럼이 빠졌는지 확인한다.
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # 3. 번역기 설정을 만든다. 실제 모델 로드/양자화/다운로드 코드는 별도 파일에 있다.
    translator = make_translator()
    # 4. 이미 번역한 원문은 캐시에서 재사용한다.
    cache = load_translation_cache(TRANSLATION_CACHE)
    success_count = 0
    error_count = 0
    skipped_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
        start = time.perf_counter()
        key = row_key(row)
        if key and complete_by_key.get(key):
            skipped_count += 1
            continue

        # 5. 짧은 description과 긴 body를 각각 정리한 뒤 하나의 원문으로 합친다.
        #    \n\n은 번역기가 둘을 다른 문단으로 볼 수 있게 하는 경계다.
        source_text = build_source_text(row)
        row_label = f"row={idx} name={str(row.get('name', '')).strip()[:80]}"
        if not source_text:
            append_output_row(row, "", 1, columns)
            if key:
                complete_by_key[key] = True
            success_count += 1
            continue

        try:
            # 6. 원문 언어를 판정하고, 영어가 아니면 원문에서 영어로 직접 번역한다.
            lang = detect_language(source_text)
            english_text = translate_source_text(translator, source_text, lang, cache, row_label)
            processed_description = build_processed_description(row, english_text)
        except Exception as exc:
            error_count += 1
            append_output_row(row, "", 0, columns)
            if key:
                complete_by_key[key] = False
            save_translation_cache(cache, TRANSLATION_CACHE)
            print(f"[row failed] index={idx} error={exc}")
            continue

        append_output_row(row, processed_description, 1, columns)
        if key:
            complete_by_key[key] = True
        success_count += 1

        if PRINT_ROW_TIMING:
            print(f"[row] index={idx} elapsed={time.perf_counter() - start:.1f}s")
        if success_count and success_count % CACHE_SAVE_EVERY == 0:
            save_translation_cache(cache, TRANSLATION_CACHE)

    # 7. 마지막 캐시를 저장하고 이번 실행의 처리 결과를 출력한다.
    save_translation_cache(cache, TRANSLATION_CACHE)
    print(
        f"done: {OUTPUT_CSV} "
        f"success={success_count} errors={error_count} skipped={skipped_count}"
    )


if __name__ == "__main__":
    main()
