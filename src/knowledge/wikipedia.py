from pathlib import Path
from urllib.parse import quote
import json
import time

import requests


# ============================================================
# AgriVision AI - Wikipedia Knowledge Connector
# ============================================================

WIKIPEDIA_API = (
    "https://en.wikipedia.org/api/rest_v1/page/summary/"
)

CACHE_FILE = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "knowledge"
    / "wikipedia_cache.json"
)

CACHE_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# EXACT 15 MODEL-CLASS MAPPING
# ============================================================

WIKIPEDIA_PAGES = {

    "Pepper__bell___Bacterial_spot":
        "Bacterial_leaf_spot_of_peppers_and_tomato",

    "Pepper__bell___healthy":
        None,

    "Potato___Early_blight":
        "Alternaria_solani",

    "Potato___Late_blight":
        "Phytophthora_infestans",

    "Potato___healthy":
        None,

    "Tomato_Bacterial_spot":
        "Bacterial_leaf_spot_of_peppers_and_tomato",

    "Tomato_Early_blight":
        "Alternaria_solani",

    "Tomato_Late_blight":
        "Phytophthora_infestans",

    "Tomato_Leaf_Mold":
        "Passalora_fulva",

    "Tomato_Septoria_leaf_spot":
        "Septoria_lycopersici",

    "Tomato_Spider_mites_Two_spotted_spider_mite":
        "Tetranychus_urticae",

    "Tomato__Target_Spot":
        "Corynespora_cassiicola",

    "Tomato__Tomato_YellowLeaf__Curl_Virus":
        "Tomato_yellow_leaf_curl_virus",

    "Tomato__Tomato_mosaic_virus":
        "Tomato_mosaic_virus",

    "Tomato_healthy":
        None,
}


# ============================================================
# CACHE
# ============================================================

def _load_cache():
    if not CACHE_FILE.exists():
        return {}

    try:
        with open(
            CACHE_FILE,
            "r",
            encoding="utf-8"
        ) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(cache):
    temporary_file = CACHE_FILE.with_suffix(
        ".tmp"
    )

    with open(
        temporary_file,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            cache,
            f,
            indent=2,
            ensure_ascii=False
        )

    temporary_file.replace(
        CACHE_FILE
    )


# ============================================================
# FETCH WIKIPEDIA
# ============================================================

def fetch_wikipedia(
    predicted_class,
    timeout=10
):
    """
    Fetch the Wikipedia summary for one of the exact
    AgriVision AI model classes.

    Returns a dictionary or None.

    Cached results are reused to avoid unnecessary
    repeated requests.
    """

    page = WIKIPEDIA_PAGES.get(
        predicted_class
    )

    if not page:
        return None

    cache = _load_cache()

    if predicted_class in cache:
        return cache[predicted_class]

    encoded_page = quote(
        page,
        safe=""
    )

    url = (
        WIKIPEDIA_API
        + encoded_page
    )

    headers = {
        "User-Agent": (
            "AgriVisionAI/1.0 "
            "(agricultural disease information application)"
        )
    }

    try:

        response = requests.get(
            url,
            headers=headers,
            timeout=timeout
        )

        response.raise_for_status()

        data = response.json()

        result = {
            "title": data.get(
                "title",
                page.replace("_", " ")
            ),
            "description": data.get(
                "description",
                ""
            ),
            "extract": data.get(
                "extract",
                ""
            ),
            "url": data.get(
                "content_urls",
                {}
            ).get(
                "desktop",
                {}
            ).get(
                "page",
                f"https://en.wikipedia.org/wiki/{page}"
            ),
            "source": "Wikipedia",
            "retrieved_at": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime()
            )
        }

        cache[predicted_class] = result
        _save_cache(cache)

        return result

    except Exception:
        return None


# ============================================================
# REPORT-FRIENDLY WIKIPEDIA SECTION
# ============================================================

def wikipedia_report_section(
    predicted_class
):
    """
    Return a Markdown section containing the Wikipedia
    information for the detected disease.

    If Wikipedia is unavailable, return a short fallback
    message without breaking the main report.
    """

    data = fetch_wikipedia(
        predicted_class
    )

    if not data:
        return ""

    title = data.get(
        "title",
        "Wikipedia"
    )

    description = data.get(
        "description",
        ""
    )

    extract = data.get(
        "extract",
        ""
    )

    url = data.get(
        "url",
        ""
    )

    section = "\n### Disease Summary\n\n"

    if extract:
        summary = " ".join(extract.split())
        sentences = [s.strip() for s in summary.split(". ") if s.strip()]
        if len(sentences) > 2:
            summary = ". ".join(sentences[:2]).strip()
            if not summary.endswith("."):
                summary += "."
        section += f"{summary}\n"

    return section
