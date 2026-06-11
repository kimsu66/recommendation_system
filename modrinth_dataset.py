"""
Modrinth Modpack Large Dataset Scraper
- Modrinth API offset 한계: 최대 10,000 (limit+offset <= 10000)
- 전략: 카테고리 × 로더 조합으로 분할 수집 후 중복 제거
- 목표: ~10,000+ unique modpacks

[CSV Columns]
- (!)name          : 모드팩 이름
- slug          : URL 식별자 (modrinth.com/modpack/{slug})
- url           : 모드팩 페이지 URL
- (!)description   : 짧은 한줄 설명
- (!)body          : 긴 상세 설명 (마크다운 → plain text 변환)
- (!)tags          : 표시용 태그 (예: Kitchen Sink, Optimization)
- (!)categories    : 카테고리 slug (예: kitchen-sink, optimization)
- loaders       : 모드 로더 (예: fabric, forge, neoforge, quilt)
- game_versions : 지원 마인크래프트 버전 (최신 5개)
- client_side   : 클라이언트 필요 여부 (required / optional / unsupported)
- server_side   : 서버 필요 여부 (required / optional / unsupported)
- license       : 라이선스 (예: MIT, ARR)
- (!)downloads     : 총 다운로드 수
- (!)followers     : 팔로워 수
- date_created  : 최초 등록일 (YYYY-MM-DD)
- date_modified : 마지막 수정일 (YYYY-MM-DD)
"""

import requests
import pandas as pd
import time
import re
from itertools import product

BASE_URL = "https://api.modrinth.com/v2"
HEADERS = {"User-Agent": "modrinth-dataset-builder/1.0 (local research)"}

# ── 수집 전략용 파라미터 ──────────────────────────────────────────────
CATEGORIES = [
    "adventure", "challenging", "combat", "kitchen-sink",
    "lightweight", "magic", "multiplayer", "optimization",
    "quests", "technology",
]
LOADERS = ["fabric", "forge", "neoforge", "quilt"]
SORT_INDEXES = ["relevance", "downloads", "follows", "newest", "updated"]

PAGE_SIZE = 100       # API 최대 100
MAX_OFFSET = 9900     # limit + offset <= 10000
DETAIL_DELAY = 0.2    # 초 (rate limit 준수)
SEARCH_DELAY = 0.3

# ── 유틸 ─────────────────────────────────────────────────────────────
def clean_markdown(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'#{1,6}\s*', '', text)
    text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)
    text = re.sub(r'`[^`]+`', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ── 검색 ─────────────────────────────────────────────────────────────
def search_modpacks(facets: list[list[str]], index: str = "relevance") -> list[dict]:
    """
    facets 조합으로 최대 10,000개 offset 범위까지 수집.
    반환: project_id 기준 중복 없는 hit list
    """
    collected = {}
    facets_str = str(facets).replace("'", '"')
    offset = 0

    while True:
        params = {
            "facets": facets_str,
            "limit": PAGE_SIZE,
            "offset": offset,
            "index": index,
        }
        try:
            resp = requests.get(f"{BASE_URL}/search", headers=HEADERS,
                                params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"    Search error (offset={offset}): {e}")
            break

        hits = data.get("hits", [])
        if not hits:
            break

        for h in hits:
            pid = h["project_id"]
            if pid not in collected:
                collected[pid] = h

        total = data.get("total_hits", 0)
        offset += len(hits)
        if offset >= min(total, MAX_OFFSET + PAGE_SIZE):
            break

        time.sleep(SEARCH_DELAY)

    return list(collected.values())


def fetch_detail(project_id: str) -> dict | None:
    try:
        resp = requests.get(f"{BASE_URL}/project/{project_id}",
                            headers=HEADERS, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"    Detail error ({project_id}): {e}")
        return None


def parse_row(hit: dict, detail: dict) -> dict | None:
    name        = hit.get("title", "").strip()
    slug        = hit.get("slug", "")
    description = hit.get("description", "").strip()
    body        = clean_markdown(detail.get("body", ""))
    downloads   = hit.get("downloads")
    followers   = hit.get("follows")
    # categories: 로더/환경 제외한 순수 카테고리
    categories  = ", ".join(hit.get("categories", []))
    # tags: display_categories (사람이 읽기 좋은 표시용 태그)
    tags        = ", ".join(hit.get("display_categories", []))
    loaders     = ", ".join(detail.get("loaders", []))
    game_versions = ", ".join(detail.get("game_versions", [])[:5])
    license_id  = detail.get("license", {}).get("id", "")
    date_created   = hit.get("date_created", "")[:10]
    date_modified  = hit.get("date_modified", "")[:10]
    client_side = detail.get("client_side", "")
    server_side = detail.get("server_side", "")
    url         = f"https://modrinth.com/modpack/{slug}"

    # 필수 컬럼 모두 있어야 row 생성
    if not all([name, description, body,
                downloads is not None, followers is not None]):
        return None

    return {
        "name":          name,
        "slug":          slug,
        "url":           url,
        "description":   description,
        "body":          body,
        "tags":          tags,
        "categories":    categories,
        "loaders":       loaders,
        "game_versions": game_versions,
        "client_side":   client_side,
        "server_side":   server_side,
        "license":       license_id,
        "downloads":     int(downloads),
        "followers":     int(followers),
        "date_created":  date_created,
        "date_modified": date_modified,
    }


# ── 메인 ─────────────────────────────────────────────────────────────
def main():
    all_hits: dict[str, dict] = {}   # project_id → hit

    # 1단계: 카테고리 없이 전체 수집 (sort 종류별)
    print("=== Phase 1: Global search by sort index ===")
    for idx in SORT_INDEXES:
        print(f"  sort={idx}")
        hits = search_modpacks(
            facets=[["project_type:modpack"]],
            index=idx,
        )
        before = len(all_hits)
        for h in hits:
            all_hits.setdefault(h["project_id"], h)
        print(f"    +{len(all_hits)-before} new  (total {len(all_hits)})")

    # 2단계: 카테고리별 수집
    print("\n=== Phase 2: Per-category search ===")
    for cat in CATEGORIES:
        for idx in ["downloads", "follows", "newest"]:
            print(f"  category={cat}, sort={idx}")
            hits = search_modpacks(
                facets=[["project_type:modpack"], [f"categories:{cat}"]],
                index=idx,
            )
            before = len(all_hits)
            for h in hits:
                all_hits.setdefault(h["project_id"], h)
            print(f"    +{len(all_hits)-before} new  (total {len(all_hits)})")

    # 3단계: 로더별 수집
    print("\n=== Phase 3: Per-loader search ===")
    for loader in LOADERS:
        for idx in ["downloads", "follows", "newest"]:
            print(f"  loader={loader}, sort={idx}")
            hits = search_modpacks(
                facets=[["project_type:modpack"], [f"categories:{loader}"]],
                index=idx,
            )
            before = len(all_hits)
            for h in hits:
                all_hits.setdefault(h["project_id"], h)
            print(f"    +{len(all_hits)-before} new  (total {len(all_hits)})")

    # 4단계: 카테고리 × 로더 조합 (아직 부족하면)
    print("\n=== Phase 4: Category × Loader combinations ===")
    for cat, loader in product(CATEGORIES, LOADERS):
        print(f"  {cat} × {loader}")
        hits = search_modpacks(
            facets=[
                ["project_type:modpack"],
                [f"categories:{cat}"],
                [f"categories:{loader}"],
            ],
            index="downloads",
        )
        before = len(all_hits)
        for h in hits:
            all_hits.setdefault(h["project_id"], h)
        if len(all_hits) - before > 0:
            print(f"    +{len(all_hits)-before} new  (total {len(all_hits)})")

    print(f"\nTotal unique modpacks collected: {len(all_hits)}")

    # 5단계: 상세 정보 수집
    print("\n=== Phase 5: Fetching details ===")
    records = []
    hits_list = list(all_hits.values())

    for i, hit in enumerate(hits_list):
        pid = hit["project_id"]
        detail = fetch_detail(pid)
        if detail is None:
            continue
        row = parse_row(hit, detail)
        if row:
            records.append(row)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(hits_list)} processed, {len(records)} valid rows")

        time.sleep(DETAIL_DELAY)

    # 6단계: 저장
    import os
    os.makedirs("./datasets", exist_ok=True)
    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["slug"])
    out = "./datasets/modrinth_dataset.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"\n✅ Done! {len(df)} rows → {out}")
    print(df[["name", "downloads", "followers", "categories"]].head(10).to_string())


if __name__ == "__main__":
    main()
