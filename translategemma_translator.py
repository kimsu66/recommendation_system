"""TranslateGemma runtime integration.

이 파일이 하는 일:
- Hugging Face에서 TranslateGemma 모델 파일을 프로젝트 `models/` 폴더로 받는다.
- 받은 모델 파일이 최소한 로드 가능한 형태인지 검사한다.
- `transformers`/`torch`/`bitsandbytes`로 모델과 processor를 로드한다.
- 긴 입력을 TranslateGemma 입력 토큰 예산에 맞춰 문단, 문장 순서로 나눈다.
- 나뉜 chunk를 직접 번역하고 번역 캐시를 만든다.

이 파일에 넣지 않는 일:
- Modrinth CSV 읽기/쓰기
- HTML/Markdown 정리
- 불용어 제거와 검색용 토큰화
- tags/categories 메타 토큰 삽입

라이브러리 사용 기준:
- TranslateGemma 모델 카드의 `AutoProcessor.apply_chat_template()`와 `generate()` 예시
- Transformers generation 문서의 greedy decoding 설정
- Transformers bitsandbytes 문서의 `BitsAndBytesConfig`
- huggingface_hub 문서의 `snapshot_download(local_dir=...)`
"""

from __future__ import annotations

import hashlib
import json
import locale
import os
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


LANG_ALIASES = {
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


class TranslationOutputLimitError(RuntimeError):
    """번역 결과가 출력 토큰 한도에 닿아 잘렸을 가능성이 있을 때 사용한다."""


@dataclass(slots=True)
class TranslateGemmaConfig:
    """TranslateGemma 실행 설정.

    preprocessor.py 상단 설정값을 이 객체에 넣어 번역기에 전달한다.
    """

    model_id: str = "google/translategemma-4b-it"
    model_dir: str | None = None
    cache_path: str = "./datasets/translation_cache.json"
    use_translation: bool = True

    quantization: str = "8bit"
    model_dtype: str = "float16"
    device_map: str | int | None = None

    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True
    bnb_8bit_threshold: float = 6.0

    max_input_tokens: int = 512
    hard_max_input_tokens: int = 1800
    max_total_tokens: int = 2048
    max_output_tokens: int = 512
    min_output_tokens: int = 64
    output_token_ratio: float = 1.3
    debug_on_failure: bool = True
    debug_to_console: bool = True
    debug_dir: str = "./logs/translation_debug"
    debug_max_chars: int = 4000
    audit_log: bool = True
    audit_dir: str = "./logs/translation_audit"
    log_progress: bool = True
    log_preview_chars: int = 120


def normalize_spacing(text: str) -> str:
    """번역 chunk를 다시 합칠 때 과도한 공백만 정리한다."""
    lines = []
    for line in str(text).replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = re.sub(r"[ \t\f\v]+", " ", line).strip()
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_lang(lang: str) -> str:
    """언어 코드를 TranslateGemma chat template이 받는 형태로 정규화한다."""
    value = str(lang or "").strip().lower().replace("_", "-")
    if not value or value == "unknown":
        return "unknown"
    return LANG_ALIASES.get(value, value.split("-", 1)[0])


def safe_model_dir_name(model_id: str) -> str:
    """Hugging Face model id를 Windows 폴더명으로 안전하게 바꾼다."""
    return re.sub(r"[^A-Za-z0-9._-]+", "__", model_id.strip())


def normalize_quantization(value: str | None) -> str:
    """사용자가 넣은 양자화 별칭을 none/8bit/4bit 중 하나로 맞춘다."""
    aliases = {
        "": "none",
        "off": "none",
        "false": "none",
        "none": "none",
        "bf16": "none",
        "fp16": "none",
        "8": "8bit",
        "int8": "8bit",
        "8-bit": "8bit",
        "8bit": "8bit",
        "4": "4bit",
        "int4": "4bit",
        "4-bit": "4bit",
        "4bit": "4bit",
        "nf4": "4bit",
    }
    quant = aliases.get(str(value or "none").strip().lower())
    if quant not in {"none", "8bit", "4bit"}:
        raise ValueError("TRANSLATE_QUANTIZATION must be one of: none, 8bit, 4bit")
    return quant


def torch_dtype(torch_module: Any, name: str):
    """문자열 dtype 설정을 torch dtype 객체로 변환한다."""
    mapping = {
        "float16": torch_module.float16,
        "fp16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "bf16": torch_module.bfloat16,
        "float32": torch_module.float32,
        "fp32": torch_module.float32,
        "auto": "auto",
    }
    key = str(name or "float16").lower()
    if key not in mapping:
        raise ValueError(f"Unknown torch dtype: {name}")
    return mapping[key]


def load_translation_cache(cache_path: str) -> dict[str, str]:
    """번역 캐시 JSON을 읽는다. 없으면 빈 dict를 반환한다."""
    path = Path(cache_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_translation_cache(cache: dict[str, str], cache_path: str) -> None:
    """번역 캐시를 디스크에 저장한다."""
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def configure_windows_subprocess_text_encoding() -> None:
    """Windows 하위 프로세스 출력 디코딩을 현재 시스템 코드페이지에 맞춘다.

    PyCharm/venv가 Python UTF-8 mode로 실행되면, `subprocess.Popen(text=True)`의
    기본 디코딩도 UTF-8이 된다. 그런데 Windows의 `cmd`, GPU/컴파일러 탐색 도구,
    일부 패키지 진단 명령은 여전히 시스템 코드페이지(cp949 등)로 출력할 수 있다.
    이 불일치가 `subprocess._readerthread`의 UnicodeDecodeError를 만든다.

    모델 계산 결과를 바꾸는 처리는 아니다. Torch/Transformers import 중 실행되는
    환경 탐색용 하위 프로세스가 Windows 출력 인코딩을 제대로 읽게 하는 보정이다.
    """
    if os.name != "nt":
        return
    if getattr(subprocess.Popen, "_modrinth_encoding_patch", False):
        return

    getencoding = getattr(locale, "getencoding", None)
    preferred_encoding = (getencoding() if getencoding else None) or locale.getpreferredencoding(False) or "mbcs"
    os.environ["PYTHONUTF8"] = "0"
    os.environ["PYTHONIOENCODING"] = f"{preferred_encoding}:replace"

    original_init = subprocess.Popen.__init__

    def patched_init(self, *args, **kwargs):
        text_mode = kwargs.get("text") or kwargs.get("universal_newlines")
        if text_mode:
            current_encoding = kwargs.get("encoding")
            if current_encoding is None:
                kwargs["encoding"] = preferred_encoding
            if kwargs.get("errors") is None:
                kwargs["errors"] = "replace"
        return original_init(self, *args, **kwargs)

    subprocess.Popen.__init__ = patched_init
    subprocess.Popen._modrinth_encoding_patch = True


class TranslateGemmaTranslator:
    """TranslateGemma 모델 로드와 번역 실행을 맡는 객체."""

    def __init__(self, config: TranslateGemmaConfig):
        self.config = config
        self._model = None
        self._processor = None

    def model_dir(self) -> Path:
        """현재 모델을 저장하거나 로드할 프로젝트 내부 폴더를 반환한다."""
        if self.config.model_dir:
            return Path(self.config.model_dir)
        return Path("./models") / safe_model_dir_name(self.config.model_id)

    def validate_local_model_files(self, model_dir: Path) -> tuple[bool, str]:
        """중단된 다운로드를 조기에 잡기 위한 최소 파일 검증.

        Hugging Face Hub는 자체 캐시에서는 etag/commit 기반으로 파일을 관리하지만,
        `local_dir`로 받은 폴더를 직접 쓰는 경우 로드 전에 index가 가리키는 shard가
        실제로 있는지 확인해 주는 편이 오류 메시지가 훨씬 명확하다.
        """
        config_json = model_dir / "config.json"
        if not config_json.exists():
            return False, "config.json not found"

        index_json = model_dir / "model.safetensors.index.json"
        if index_json.exists():
            try:
                index = json.loads(index_json.read_text(encoding="utf-8"))
                required = sorted(set(index.get("weight_map", {}).values()))
            except json.JSONDecodeError as exc:
                return False, f"model.safetensors.index.json is invalid: {exc}"

            missing = [name for name in required if not (model_dir / name).exists()]
            empty = [name for name in required if (model_dir / name).exists() and (model_dir / name).stat().st_size == 0]
            if missing or empty:
                return False, f"missing={missing[:3]}, empty={empty[:3]}"
            return True, "ok"

        safetensors = list(model_dir.glob("*.safetensors"))
        if not safetensors:
            return False, "no safetensors weights found"
        empty = [path.name for path in safetensors if path.stat().st_size == 0]
        if empty:
            return False, f"empty safetensors files: {empty[:3]}"
        return True, "ok"

    def ensure_model_dir(self) -> str:
        """모델 폴더를 준비하고, 불완전하면 다시 다운로드를 시도한다."""
        configure_windows_subprocess_text_encoding()

        model_dir = self.model_dir()
        is_valid, reason = self.validate_local_model_files(model_dir) if model_dir.exists() else (False, "not downloaded")
        if is_valid:
            return str(model_dir)

        from huggingface_hub import snapshot_download

        print(f"[translator] model download/repair: {self.config.model_id} ({reason})")
        model_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=self.config.model_id, local_dir=str(model_dir))

        is_valid, reason = self.validate_local_model_files(model_dir)
        if not is_valid:
            raise RuntimeError(f"TranslateGemma model download is incomplete: {model_dir} ({reason})")
        return str(model_dir)

    def build_quantization_config(self, torch_module: Any):
        """Transformers `from_pretrained`에 넘길 BitsAndBytesConfig를 만든다."""
        quant = normalize_quantization(self.config.quantization)
        if quant == "none":
            return None

        from transformers import BitsAndBytesConfig

        if quant == "8bit":
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=self.config.bnb_8bit_threshold,
            )

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=torch_dtype(torch_module, self.config.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
        )

    def effective_device_map(self, torch_module: Any):
        """기본 장치 배치.

        CUDA가 있으면 한 GPU에 전부 올린다. 이 설정은 `auto` offload 때문에
        4bit 모델이 CPU/디스크로 밀리며 실패하는 상황을 피하기 위한 기본값이다.
        """
        if self.config.device_map is not None:
            return self.config.device_map
        return 0 if torch_module.cuda.is_available() else "cpu"

    def load(self):
        """TranslateGemma processor와 모델을 한 번만 로드한다."""
        if not self.config.use_translation:
            return None, None
        if self._model is not None:
            return self._model, self._processor

        configure_windows_subprocess_text_encoding()

        import torch
        from transformers import AutoProcessor

        model_dir = self.ensure_model_dir()
        model_kwargs = {
            "device_map": self.effective_device_map(torch),
            "dtype": torch_dtype(torch, self.config.model_dtype),
            "local_files_only": True,
        }
        quantization_config = self.build_quantization_config(torch)
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config

        self._processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)

        from transformers import AutoModelForImageTextToText

        self._model = AutoModelForImageTextToText.from_pretrained(model_dir, **model_kwargs)

        print("[translator]")
        print(f"- model: {self.config.model_id}")
        print("- model_loader: AutoModelForImageTextToText")
        print(f"- quantization: {normalize_quantization(self.config.quantization)}")
        print(f"- generation_pad_token_id: {self._model.generation_config.pad_token_id}")
        print(f"- generation_eos_token_id: {self._model.generation_config.eos_token_id}")
        if hasattr(self._model, "get_memory_footprint"):
            print(f"- memory: {self._model.get_memory_footprint() / (1024 ** 3):.2f} GiB")
        if torch.cuda.is_available():
            print(f"- cuda_allocated: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GiB")

        return self._model, self._processor

    def translation_messages(self, text: str, source_lang: str, target_lang: str) -> list[dict[str, Any]]:
        """TranslateGemma chat template 입력을 만든다."""
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": normalize_lang(source_lang),
                        "target_lang_code": normalize_lang(target_lang),
                        "text": text,
                    }
                ],
            }
        ]

    def count_input_tokens(self, text: str, source_lang: str, target_lang: str) -> int:
        """chat template 적용 후 실제 입력 토큰 수를 계산한다."""
        _model, processor = self.load()
        inputs = processor.apply_chat_template(
            self.translation_messages(text, source_lang, target_lang),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        return int(inputs["input_ids"].shape[-1])

    def split_paragraphs(self, text: str) -> list[str]:
        """빈 줄 기준 문단 분리. 문맥 유지를 위해 최우선 분할 단위로 쓴다."""
        return [p.strip() for p in re.split(r"\n\s*\n+", str(text)) if p.strip()]

    def split_sentences(self, text: str) -> list[str]:
        """단일 문단이 토큰 예산을 넘을 때만 문장 단위로 나눈다."""
        pieces = re.split(r"(?<=[.!?。！？])\s+", str(text).strip())
        return [piece.strip() for piece in pieces if piece.strip()] or [str(text).strip()]

    def chunk_input_token_limit(self) -> int:
        """번역 출력까지 고려해 모델 카드의 2K 예산 안에 들어갈 입력 chunk 크기를 잡는다."""
        ratio_limit = int((self.config.max_total_tokens - self.config.min_output_tokens) / (1 + self.config.output_token_ratio))
        return max(1, min(self.config.max_input_tokens, ratio_limit))

    def pack_segments(self, segments: list[str], separator: str, source_lang: str, target_lang: str) -> list[str]:
        """문단/문장을 입력 토큰 예산 이하의 chunk 목록으로 묶는다."""
        chunks: list[str] = []
        current = ""
        token_limit = self.chunk_input_token_limit()

        for segment in segments:
            if self.count_input_tokens(segment, source_lang, target_lang) > token_limit:
                if current:
                    chunks.append(current)
                    current = ""
                if separator == " ":
                    if self.count_input_tokens(segment, source_lang, target_lang) <= self.config.hard_max_input_tokens:
                        print(f"[translation oversized sentence] input_tokens>{token_limit}; trying as one chunk")
                        chunks.append(segment)
                        continue
                    print(f"[translation skip: over hard token budget] {segment[:120]}")
                    continue
                chunks.extend(self.pack_segments(self.split_sentences(segment), " ", source_lang, target_lang))
                continue

            candidate = segment if not current else f"{current}{separator}{segment}"
            if self.count_input_tokens(candidate, source_lang, target_lang) <= token_limit:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = segment

        if current:
            chunks.append(current)
        return chunks

    def split_for_translation(self, text: str, source_lang: str, target_lang: str) -> list[str]:
        """전체 원문을 문단 우선, 문장 fallback 방식으로 chunk화한다."""
        return self.pack_segments(self.split_paragraphs(text), "\n\n", source_lang, target_lang)

    def preview_text(self, text: str) -> str:
        """진행 로그에 넣을 짧은 미리보기 문자열을 만든다."""
        text = normalize_spacing(text).replace("\n", " ")
        limit = self.config.log_preview_chars
        if limit and len(text) > limit:
            return text[:limit] + "..."
        return text

    def token_stats(self, token_counts: list[int]) -> str:
        """chunk 토큰 수 요약을 사람이 읽기 좋게 만든다."""
        if not token_counts:
            return "min=0 avg=0 max=0"
        avg = sum(token_counts) / len(token_counts)
        return f"min={min(token_counts)} avg={avg:.1f} max={max(token_counts)}"

    def char_stats(self, chunks: list[str]) -> str:
        """chunk 문자 수 요약을 사람이 읽기 좋게 만든다."""
        if not chunks:
            return "min=0 avg=0 max=0"
        lengths = [len(chunk) for chunk in chunks]
        avg = sum(lengths) / len(lengths)
        return f"min={min(lengths)} avg={avg:.1f} max={max(lengths)}"

    def max_new_token_cap(self, input_tokens: int) -> int:
        """입력+출력이 모델 카드의 2K 예산을 넘지 않도록 출력 상한을 계산한다."""
        remaining = self.config.max_total_tokens - input_tokens
        return max(self.config.min_output_tokens, min(self.config.max_output_tokens, remaining))

    def max_new_tokens_for(self, input_tokens: int) -> int:
        """입력 길이에 맞춰 chunk 하나의 기본 출력 토큰 상한을 잡는다."""
        budget = int(input_tokens * self.config.output_token_ratio)
        budget = max(self.config.min_output_tokens, budget)
        return min(budget, self.max_new_token_cap(input_tokens))

    def clipped_debug_text(self, text: str) -> str:
        """콘솔에 출력할 디버그 텍스트를 제한한다."""
        text = str(text)
        limit = self.config.debug_max_chars
        if limit and len(text) > limit:
            return text[:limit] + f"\n...[truncated debug text: {len(text) - limit} chars omitted]"
        return text

    def write_translation_debug(
        self,
        reason: str,
        source_lang: str,
        target_lang: str,
        chunk: str,
        prompt_text: str,
        generated_raw: str,
        generated_clean: str,
        input_tokens: int,
        output_tokens: int,
        max_new_tokens: int,
        eos_token_ids,
    ) -> None:
        """번역 실패 시 실제 입력/프롬프트/생성 결과를 화면과 파일에 남긴다."""
        if not self.config.debug_on_failure:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        digest = hashlib.sha1(f"{reason}\n{source_lang}\n{target_lang}\n{chunk}".encode("utf-8")).hexdigest()[:12]
        debug_path = Path(self.config.debug_dir) / f"{timestamp}_{digest}.txt"
        debug_path.parent.mkdir(parents=True, exist_ok=True)

        content = "\n".join([
            "=== TRANSLATION DEBUG ===",
            f"reason: {reason}",
            f"source_lang: {source_lang}",
            f"target_lang: {target_lang}",
            f"input_tokens: {input_tokens}",
            f"output_tokens: {output_tokens}",
            f"max_new_tokens: {max_new_tokens}",
            f"eos_token_ids: {eos_token_ids}",
            "",
            "=== SOURCE CHUNK ===",
            chunk,
            "",
            "=== RENDERED PROMPT ===",
            prompt_text,
            "",
            "=== GENERATED RAW / SPECIAL TOKENS KEPT ===",
            generated_raw,
            "",
            "=== GENERATED CLEAN / SPECIAL TOKENS REMOVED ===",
            generated_clean,
            "",
        ])
        debug_path.write_text(content, encoding="utf-8")

        if self.config.debug_to_console:
            print("\n[translation debug]")
            print(f"- reason: {reason}")
            print(f"- source={source_lang} target={target_lang}")
            print(f"- input_tokens={input_tokens} output_tokens={output_tokens} max_new_tokens={max_new_tokens}")
            print(f"- eos_token_ids={eos_token_ids}")
            print(f"- debug_file: {debug_path}")
            print("[source chunk]")
            print(self.clipped_debug_text(chunk))
            print("[generated clean]")
            print(self.clipped_debug_text(generated_clean))
            print("[generated raw]")
            print(self.clipped_debug_text(generated_raw))

    def write_translation_audit(
        self,
        source_lang: str,
        target_lang: str,
        label: str,
        source_text: str,
        translated_text: str,
        chunks: list[str],
        translated_chunks: list[str],
        total_elapsed: float,
        cache_hit: bool = False,
    ) -> None:
        """번역 입출력을 항상 파일로 남겨 언어 감지와 번역 품질을 수동 검수할 수 있게 한다."""
        if not self.config.audit_log:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        digest = hashlib.sha1(f"{source_lang}\n{target_lang}\n{label}\n{source_text}".encode("utf-8")).hexdigest()[:12]
        audit_path = Path(self.config.audit_dir) / f"{timestamp}_{digest}.txt"
        audit_path.parent.mkdir(parents=True, exist_ok=True)

        content = [
            "=== TRANSLATION AUDIT ===",
            f"label: {label}",
            f"source_lang: {source_lang}",
            f"target_lang: {target_lang}",
            f"cache_hit: {int(cache_hit)}",
            f"chunks: {len(chunks)}",
            f"elapsed_sec: {total_elapsed:.1f}",
            f"source_chars: {len(source_text)}",
            f"translated_chars: {len(translated_text)}",
            "",
            "=== SOURCE TEXT ===",
            source_text,
            "",
            "=== TRANSLATED TEXT ===",
            translated_text,
            "",
        ]
        for index, (chunk, translated_chunk) in enumerate(zip(chunks, translated_chunks), 1):
            content.extend([
                f"=== CHUNK {index}/{len(chunks)} SOURCE ===",
                chunk,
                "",
                f"=== CHUNK {index}/{len(chunks)} TRANSLATED ===",
                translated_chunk,
                "",
            ])
        audit_path.write_text("\n".join(content), encoding="utf-8")
        if self.config.log_progress:
            print(f"[translation audit] {audit_path}")

    def translate_chunk_once(self, chunk: str, source_lang: str, target_lang: str, max_new_tokens: int) -> str:
        """chunk 하나를 지정된 출력 토큰 한도로 한 번 번역한다."""
        import torch

        model, processor = self.load()
        inputs = processor.apply_chat_template(
            self.translation_messages(chunk, source_lang, target_lang),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_len = int(inputs["input_ids"].shape[-1])
        eos_token_ids = getattr(model.generation_config, "eos_token_id", None)
        prompt_text = processor.decode(inputs["input_ids"][0], skip_special_tokens=False)

        inputs = inputs.to(model.device)

        generate_kwargs = {
            **inputs,
            "do_sample": False,
            "max_new_tokens": max_new_tokens,
        }

        with torch.inference_mode():
            output = model.generate(**generate_kwargs)

        generated = output[0][input_len:]
        generated_raw = processor.decode(generated, skip_special_tokens=False).strip()
        generated_clean = processor.decode(generated, skip_special_tokens=True).strip()
        if generated_raw and not generated_clean and generated_raw.replace("<pad>", "").strip() == "":
            self.write_translation_debug(
                reason="pad_only_output",
                source_lang=source_lang,
                target_lang=target_lang,
                chunk=chunk,
                prompt_text=prompt_text,
                generated_raw=generated_raw,
                generated_clean=generated_clean,
                input_tokens=input_len,
                output_tokens=len(generated),
                max_new_tokens=max_new_tokens,
                eos_token_ids=eos_token_ids,
            )
            raise RuntimeError("translation generated only <pad> tokens")
        if len(generated) >= max_new_tokens:
            self.write_translation_debug(
                reason="output_token_limit_hit",
                source_lang=source_lang,
                target_lang=target_lang,
                chunk=chunk,
                prompt_text=prompt_text,
                generated_raw=generated_raw,
                generated_clean=generated_clean,
                input_tokens=input_len,
                output_tokens=len(generated),
                max_new_tokens=max_new_tokens,
                eos_token_ids=eos_token_ids,
            )
            raise TranslationOutputLimitError(
                f"translation output token limit hit: input_tokens={input_len}, max_new_tokens={max_new_tokens}"
            )
        return generated_clean

    def split_chunk_for_output_retry(self, chunk: str) -> list[str]:
        """출력 한도에 걸린 chunk를 더 작은 단위로 나눈다."""
        paragraphs = self.split_paragraphs(chunk)
        if len(paragraphs) > 1:
            return paragraphs

        sentences = self.split_sentences(chunk)
        if len(sentences) > 1:
            return sentences

        # 마지막 fallback이다. 단일 문장이 너무 길어 번역 출력이 계속 잘리면
        # 쉼표/세미콜론 같은 약한 구분점으로 쪼개 본다.
        clauses = [part.strip() for part in re.split(r"(?<=[,;:])\s+", chunk) if part.strip()]
        if len(clauses) > 1:
            return clauses

        return [chunk]

    def translate_chunk(self, chunk: str, source_lang: str, target_lang: str, depth: int = 0) -> str:
        """chunk 하나를 번역한다.

        출력 한도에 걸리면 `max_output_tokens`까지 한 번 올려 재시도하고,
        그래도 잘리면 chunk를 더 작게 나눠 재귀적으로 다시 번역한다.
        """
        input_tokens = self.count_input_tokens(chunk, source_lang, target_lang)
        if input_tokens > self.config.hard_max_input_tokens:
            raise RuntimeError(
                f"translation input hard limit exceeded: input_tokens={input_tokens}, "
                f"hard_max_input_tokens={self.config.hard_max_input_tokens}"
            )

        first_limit = self.max_new_tokens_for(input_tokens)
        retry_limits = [first_limit]
        retry_limit = self.max_new_token_cap(input_tokens)
        if retry_limit not in retry_limits:
            retry_limits.append(retry_limit)

        last_error: Exception | None = None
        for limit in retry_limits:
            try:
                return self.translate_chunk_once(chunk, source_lang, target_lang, limit)
            except TranslationOutputLimitError as exc:
                last_error = exc
                if limit < retry_limit:
                    print(f"[translation retry: larger output] {limit}->{retry_limit}")
                    continue

        smaller_chunks = self.split_chunk_for_output_retry(chunk)
        if len(smaller_chunks) > 1 and depth < 3:
            print(f"[translation retry: split smaller] {len(smaller_chunks)} chunks")
            translated = [
                self.translate_chunk(part, source_lang, target_lang, depth=depth + 1)
                for part in smaller_chunks
                if part.strip()
            ]
            return normalize_spacing("\n\n".join(part for part in translated if part))

        raise last_error or RuntimeError("translation failed")

    def cache_key(self, text: str, source_lang: str, target_lang: str) -> str:
        """모델/토큰 정책이 바뀌면 달라지는 번역 캐시 키를 만든다."""
        raw = {
            "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "source_lang": normalize_lang(source_lang),
            "target_lang": normalize_lang(target_lang),
            "model_id": self.config.model_id,
            "model_loader": "AutoModelForImageTextToText",
            "quantization": normalize_quantization(self.config.quantization),
            "max_input_tokens": self.config.max_input_tokens,
            "hard_max_input_tokens": self.config.hard_max_input_tokens,
            "max_total_tokens": self.config.max_total_tokens,
            "max_output_tokens": self.config.max_output_tokens,
            "chunking": "paragraph-then-sentence-v1",
            "generation_stop": "official-generate-config-v2-context-budget",
        }
        return hashlib.sha256(json.dumps(raw, sort_keys=True).encode("utf-8")).hexdigest()

    def translate_text(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        cache: dict[str, str],
        label: str = "",
        raise_on_failure: bool = False,
    ) -> str:
        """긴 텍스트를 chunk 단위로 직접 번역하고 캐시한다."""
        started_at = time.perf_counter()
        text = str(text).strip()
        source_lang = normalize_lang(source_lang)
        target_lang = normalize_lang(target_lang)
        label_text = f" {label}" if label else ""

        if not text or not self.config.use_translation:
            return text if source_lang == target_lang else ""
        if source_lang == target_lang:
            return text
        if source_lang == "unknown":
            raise ValueError("TranslateGemma requires a known source language code")

        key = self.cache_key(text, source_lang, target_lang)
        if key in cache:
            if self.config.log_progress:
                print(f"[translation cache hit]{label_text} {source_lang}->{target_lang} chars={len(text)}")
            self.write_translation_audit(
                source_lang=source_lang,
                target_lang=target_lang,
                label=label,
                source_text=text,
                translated_text=cache[key],
                chunks=[text],
                translated_chunks=[cache[key]],
                total_elapsed=time.perf_counter() - started_at,
                cache_hit=True,
            )
            return cache[key]

        paragraphs = self.split_paragraphs(text)
        paragraph_tokens = [self.count_input_tokens(paragraph, source_lang, target_lang) for paragraph in paragraphs]
        oversized_paragraphs = sum(tokens > self.config.max_input_tokens for tokens in paragraph_tokens)

        if self.config.log_progress:
            print(
                f"[translation start]{label_text} {source_lang}->{target_lang} "
                f"chars={len(text)} paragraphs={len(paragraphs)} "
                f"paragraph_tokens=({self.token_stats(paragraph_tokens)}) "
                f"oversized_paragraphs={oversized_paragraphs} "
                f"preview={self.preview_text(text)}"
            )

        chunks = self.pack_segments(paragraphs, "\n\n", source_lang, target_lang)
        chunk_tokens = [self.count_input_tokens(chunk, source_lang, target_lang) for chunk in chunks]

        if self.config.log_progress:
            print(
                f"[translation split]{label_text} {source_lang}->{target_lang} "
                f"chunks={len(chunks)} chunk_tokens=({self.token_stats(chunk_tokens)}) "
                f"chunk_chars=({self.char_stats(chunks)})"
            )

        translated: list[str] = []
        failed_chunks = 0
        for index, chunk in enumerate(chunks, 1):
            start = time.perf_counter()
            input_tokens = chunk_tokens[index - 1] if index - 1 < len(chunk_tokens) else self.count_input_tokens(chunk, source_lang, target_lang)
            max_new_tokens = self.max_new_tokens_for(input_tokens)
            if self.config.log_progress:
                print(
                    f"[translation chunk start]{label_text} {source_lang}->{target_lang} "
                    f"chunk={index}/{len(chunks)} chars={len(chunk)} "
                    f"input_tokens={input_tokens} max_new_tokens={max_new_tokens} "
                    f"preview={self.preview_text(chunk)}"
                )
            try:
                result = self.translate_chunk(chunk, source_lang, target_lang)
            except Exception as exc:
                failed_chunks += 1
                print(
                    "[translation chunk failed] "
                    f"{index}/{len(chunks)} source={source_lang} target={target_lang} "
                    f"elapsed={time.perf_counter() - start:.1f}s error={exc}"
                )
                continue
            elapsed = time.perf_counter() - start
            output_tokens_estimate = len(result.split())
            print(
                f"[translation chunk done]{label_text} {source_lang}->{target_lang} "
                f"chunk={index}/{len(chunks)} elapsed={elapsed:.1f}s "
                f"output_chars={len(result)} output_words~={output_tokens_estimate}"
            )
            if result:
                translated.append(result)

        result_text = normalize_spacing("\n\n".join(translated))
        total_elapsed = time.perf_counter() - started_at
        if failed_chunks:
            print(
                f"[translation done with failures]{label_text} {source_lang}->{target_lang} "
                f"elapsed={total_elapsed:.1f}s chunks={len(chunks)} failed={failed_chunks} "
                f"output_chars={len(result_text)} cache_saved=False"
            )
            if raise_on_failure:
                raise RuntimeError(
                    f"translation incomplete: chunks={len(chunks)} failed={failed_chunks} "
                    f"source={source_lang} target={target_lang}"
                )
            return result_text

        cache[key] = result_text
        self.write_translation_audit(
            source_lang=source_lang,
            target_lang=target_lang,
            label=label,
            source_text=text,
            translated_text=result_text,
            chunks=chunks,
            translated_chunks=translated,
            total_elapsed=total_elapsed,
            cache_hit=False,
        )
        if self.config.log_progress:
            print(
                f"[translation done]{label_text} {source_lang}->{target_lang} "
                f"elapsed={total_elapsed:.1f}s chunks={len(chunks)} failed=0 "
                f"output_chars={len(result_text)} cache_saved=True"
            )
        return result_text
