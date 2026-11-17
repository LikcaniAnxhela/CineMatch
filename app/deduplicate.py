"""Data cleaning and deduplication for the CineMatch movie catalogue.

The raw ``data/movies.json`` file is treated as immutable source data.  Every
write performed by this module targets ``data/movies_clean.json`` (or an
explicit path supplied by the caller), never the original file.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
SOURCE_PATH = DATA_DIR / "movies.json"
CLEAN_PATH = DATA_DIR / "movies_clean.json"

_WHITESPACE_RE = re.compile(r"\s+")


def _as_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return _WHITESPACE_RE.sub(" ", str(value)).strip()


def _as_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clean_string_list(value: Any, *, lowercase: bool = False) -> List[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    cleaned: List[str] = []
    seen: set = set()
    for item in value:
        text = _as_text(item)
        if not text:
            continue
        if lowercase:
            text = text.lower()
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
    return cleaned


def normalize_text(value: Any) -> str:
    """Normalise text for fuzzy comparisons (case/accent/punctuation safe)."""
    text = _as_text(value).casefold()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def normalize_title(value: Any) -> str:
    return normalize_text(value)


def clean_movie(movie: Any) -> Optional[dict]:
    """Return a normalised copy of ``movie`` or ``None`` when unusable."""
    if not isinstance(movie, dict):
        return None

    movie_id = _as_int(movie.get("id"))
    title = _as_text(movie.get("title"))
    if movie_id is None or not title:
        return None

    genres = _clean_string_list(movie.get("genres"))
    keywords = _clean_string_list(movie.get("keywords"), lowercase=True)
    vote_average = max(0.0, min(10.0, _as_float(movie.get("vote_average"), 0.0)))

    return {
        "id": movie_id,
        "title": title,
        "original_title": _as_text(movie.get("original_title")) or title,
        "year": _as_int(movie.get("year")),
        "runtime": _as_int(movie.get("runtime")),
        "genres": genres,
        "keywords": keywords,
        "overview": _as_text(movie.get("overview")),
        "tagline": _as_text(movie.get("tagline")),
        "vote_average": round(vote_average, 2),
        "vote_count": max(0, _as_int(movie.get("vote_count"), 0) or 0),
        "popularity": round(max(0.0, _as_float(movie.get("popularity"), 0.0)), 3),
        "language": _as_text(movie.get("language")) or "en",
    }


def _quality_key(movie: dict) -> Tuple[int, int, int, int]:
    """Rank duplicate candidates; higher is better and deterministic."""
    return (
        movie.get("vote_count") or 0,
        len(movie.get("keywords") or []),
        len(movie.get("genres") or []),
        len(movie.get("overview") or ""),
    )


def deduplicate_movies(movies: Iterable[dict]) -> List[dict]:
    """Clean, validate and deduplicate movies without mutating the input.

    Two passes are applied:
    1. Duplicate IDs keep the highest-quality record.
    2. Duplicate normalised ``(title, year)`` pairs keep the highest-quality
       record.  This catches the real catalogue collisions such as multiple
       ``Interstellar (2014)`` entries.
    """
    prepared: List[Tuple[int, dict]] = []
    for index, raw in enumerate(movies):
        cleaned = clean_movie(raw)
        if cleaned is not None:
            prepared.append((index, cleaned))

    by_id: Dict[int, Tuple[int, dict]] = {}
    for index, movie in prepared:
        existing = by_id.get(movie["id"])
        if existing is None or _quality_key(movie) > _quality_key(existing[1]):
            by_id[movie["id"]] = (index, movie)

    by_title: Dict[Tuple[str, Optional[int]], Tuple[int, dict]] = {}
    for index, movie in by_id.values():
        key = (normalize_title(movie["title"]), movie.get("year"))
        existing = by_title.get(key)
        if existing is None or _quality_key(movie) > _quality_key(existing[1]):
            by_title[key] = (index, movie)

    chosen = sorted(by_title.values(), key=lambda item: item[0])
    return [movie for _, movie in chosen]


def clean_movies(movies: Iterable[dict]) -> List[dict]:
    """Alias with a clearer name for the full cleaning pipeline."""
    return deduplicate_movies(movies)


def catalogue_stats(movies: Iterable[dict]) -> dict:
    movies = list(movies)
    genres: Dict[str, int] = {}
    years: List[int] = []
    for movie in movies:
        for genre in movie.get("genres") or []:
            genres[genre] = genres.get(genre, 0) + 1
        year = movie.get("year")
        if isinstance(year, int) and year > 0:
            years.append(year)
    return {
        "movie_count": len(movies),
        "genre_count": len(genres),
        "missing_year": sum(1 for m in movies if not m.get("year")),
        "missing_genres": sum(1 for m in movies if not m.get("genres")),
        "missing_keywords": sum(1 for m in movies if not m.get("keywords")),
        "year_min": min(years) if years else None,
        "year_max": max(years) if years else None,
        "top_genres": sorted(genres.items(), key=lambda item: item[1], reverse=True),
    }


def load_movies(path: Optional[Path] = None) -> List[dict]:
    movies_path = Path(path) if path is not None else SOURCE_PATH
    if not movies_path.exists():
        return []
    with open(movies_path, "r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    return data if isinstance(data, list) else []


def load_and_deduplicate(path: Optional[Path] = None) -> List[dict]:
    """Load a JSON catalogue and return the cleaned/deduplicated list."""
    return deduplicate_movies(load_movies(path))


def save_cleaned_movies(
    movies: Optional[Iterable[dict]] = None,
    path: Optional[Path] = None,
    source_path: Optional[Path] = None,
) -> str:
    """Persist cleaned records to ``movies_clean.json`` (never the source)."""
    cleaned = list(movies) if movies is not None else load_and_deduplicate(source_path)
    movies_path = Path(path) if path is not None else CLEAN_PATH
    movies_path.parent.mkdir(parents=True, exist_ok=True)
    with open(movies_path, "w", encoding="utf-8") as handle:
        json.dump(cleaned, handle, ensure_ascii=False, indent=2)
    return str(movies_path)


def classify_movie_keywords(movies: Iterable[dict]) -> Dict[str, List[int]]:
    """Map each lowercase keyword to the IDs of movies carrying it."""
    keyword_map: Dict[str, List[int]] = {}
    for movie in movies:
        mid = movie.get("id")
        if mid is None:
            continue
        for keyword in movie.get("keywords") or []:
            key = _as_text(keyword).casefold()
            if not key:
                continue
            bucket = keyword_map.setdefault(key, [])
            if mid not in bucket:
                bucket.append(mid)
    return keyword_map


def classify_movie_genres(movies: Iterable[dict]) -> Dict[str, List[int]]:
    """Map each lowercase genre to the IDs of movies carrying it."""
    genre_map: Dict[str, List[int]] = {}
    for movie in movies:
        mid = movie.get("id")
        if mid is None:
            continue
        for genre in movie.get("genres") or []:
            key = _as_text(genre).casefold()
            if not key:
                continue
            bucket = genre_map.setdefault(key, [])
            if mid not in bucket:
                bucket.append(mid)
    return genre_map
