"""CineMatch recommendation engines.

Two intentionally different strategies are exposed:

* ``naive_recommend`` (v1) - a transparent, flat genre/keyword overlap score
  with an unweighted vote-average boost.
* ``smart_recommend`` (v2) - a rating-weighted taste profile, IDF keyword
  affinity, Bayesian rating smoothing and greedy genre diversification.

Both engines use a user's favourite genres *and* their movie ratings, work
fully offline and are deterministic for a given catalogue/profile.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
SOURCE_PATH = DATA_DIR / "movies.json"
CLEAN_PATH = DATA_DIR / "movies_clean.json"

DEFAULT_TOP_N = 12
BAYESIAN_MIN_VOTES = 80


def get_movies() -> List[dict]:
    """Load the cleaned catalogue when available, otherwise clean on the fly."""
    if CLEAN_PATH.exists():
        with open(CLEAN_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list) and data:
            return data

    if SOURCE_PATH.exists():
        from .deduplicate import load_and_deduplicate

        return load_and_deduplicate(SOURCE_PATH)

    return []


def _key(value: object) -> str:
    return str(value).strip().casefold()


def _genres(movie: dict) -> List[str]:
    return [str(g) for g in (movie.get("genres") or []) if str(g).strip()]


def _keywords(movie: dict) -> List[str]:
    return [str(k) for k in (movie.get("keywords") or []) if str(k).strip()]


def _rating(movie: dict) -> float:
    try:
        return float(movie.get("vote_average") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _votes(movie: dict) -> int:
    try:
        return max(0, int(movie.get("vote_count") or 0))
    except (TypeError, ValueError):
        return 0


def _year(movie: dict) -> int:
    try:
        return int(movie.get("year") or 0)
    except (TypeError, ValueError):
        return 0


def _popularity(movie: dict) -> float:
    try:
        return max(0.0, float(movie.get("popularity") or 0.0))
    except (TypeError, ValueError):
        return 0.0


def build_user_profile(
    movies: Iterable[dict],
    user_genres: Optional[Iterable[str]],
    user_rated: Optional[Dict[int, float]],
) -> dict:
    """Build a taste profile from explicit favourites and rated movies.

    Ratings act as confidence weights (1 star = 0.2, 5 stars = 1.0).  Positive
    preferences are emphasised, which keeps low ratings from pulling the
    profile too far toward disliked content while still providing signal.
    """
    by_id = {movie.get("id"): movie for movie in movies}
    genre_weights: Counter = Counter()
    keyword_weights: Counter = Counter()
    rated_ids: set = set()
    rated_movies: List[dict] = []

    for raw_id, raw_rating in (user_rated or {}).items():
        try:
            movie_id = int(raw_id)
            rating = max(1.0, min(5.0, float(raw_rating)))
        except (TypeError, ValueError):
            continue
        movie = by_id.get(movie_id)
        if movie is None:
            continue
        rated_ids.add(movie_id)
        rated_movies.append(movie)
        weight = rating / 5.0
        for genre in _genres(movie):
            genre_weights[_key(genre)] += weight
        for keyword in _keywords(movie):
            keyword_weights[_key(keyword)] += weight

    for genre in user_genres or []:
        if _key(genre):
            genre_weights[_key(genre)] += 1.75

    return {
        "genre_weights": genre_weights,
        "keyword_weights": keyword_weights,
        "rated_ids": rated_ids,
        "rated_movies": rated_movies,
        "favourite_genres": [_key(g) for g in (user_genres or []) if _key(g)],
    }


def _catalogue_context(movies: List[dict]) -> dict:
    """Pre-compute IDF weights and Bayesian rating priors for smart ranking."""
    total = max(1, len(movies))
    keyword_freq: Counter = Counter()
    rating_sum = 0.0
    rating_votes = 0
    max_popularity = 0.0

    for movie in movies:
        for keyword in set(_keywords(movie)):
            keyword_freq[_key(keyword)] += 1
        votes = _votes(movie)
        rating_sum += _rating(movie) * votes
        rating_votes += votes
        max_popularity = max(max_popularity, _popularity(movie))

    mean_rating = (rating_sum / rating_votes) if rating_votes else 6.0
    idf = {keyword: math.log((total + 1) / (freq + 1)) + 1.0 for keyword, freq in keyword_freq.items()}
    return {
        "mean_rating": mean_rating,
        "max_popularity": max_popularity or 1.0,
        "idf": idf,
        "total": total,
    }


def _bayesian_rating(movie: dict, mean_rating: float, minimum_votes: int = BAYESIAN_MIN_VOTES) -> float:
    votes = _votes(movie)
    return ((_rating(movie) * votes) + (mean_rating * minimum_votes)) / (votes + minimum_votes)


def _recency_bonus(year: int, newest_year: int) -> float:
    if not year or not newest_year:
        return 0.0
    span = max(1, newest_year - 1900)
    return max(0.0, min(1.0, (year - 1900) / span))


def _result(movie: dict, score: float, reason: str) -> dict:
    return {
        "id": movie.get("id"),
        "title": movie.get("title", "Unknown"),
        "year": _year(movie) or None,
        "rating": round(_rating(movie), 2),
        "vote_count": _votes(movie),
        "genres": _genres(movie)[:4],
        "score": round(score, 4),
        "reason": reason,
    }


def _fallback_order(movies: List[dict], smart: bool, context: Optional[dict] = None) -> List[dict]:
    if smart:
        context = context or _catalogue_context(movies)
        ordered = sorted(
            movies,
            key=lambda m: (
                _bayesian_rating(m, context["mean_rating"]),
                _votes(m),
                _popularity(m),
            ),
            reverse=True,
        )
    else:
        ordered = sorted(movies, key=lambda m: (_rating(m), _votes(m), _popularity(m)), reverse=True)
    return ordered


def naive_recommend(
    user_genres: List[str],
    user_rated: Dict[int, float],
    movies: Optional[List[dict]] = None,
    top_n: int = DEFAULT_TOP_N,
) -> List[dict]:
    """v1: flat, unweighted overlap of genres/keywords plus raw vote average."""
    movies = list(movies) if movies is not None else get_movies()
    top_n = max(0, int(top_n))
    if not movies or top_n == 0:
        return []

    profile = build_user_profile(movies, user_genres, user_rated)
    if not profile["genre_weights"] and not profile["keyword_weights"]:
        ordered = _fallback_order(movies, smart=False)
        return [_result(m, _rating(m) + _votes(m) / 100_000, "Popular with viewers") for m in ordered[:top_n]]

    profile_genres = set(profile["genre_weights"])
    profile_keywords = set(profile["keyword_weights"])
    rated_ids = profile["rated_ids"]
    newest_year = max((_year(m) for m in movies), default=0)
    max_popularity = max((_popularity(m) for m in movies), default=1.0) or 1.0

    scored: List[dict] = []
    for movie in movies:
        if movie.get("id") in rated_ids:
            continue

        genre_hits = [g for g in _genres(movie) if _key(g) in profile_genres]
        keyword_hits = [k for k in _keywords(movie) if _key(k) in profile_keywords]
        score = (
            2.4 * len(genre_hits)
            + 0.7 * len(keyword_hits)
            + 0.55 * _rating(movie)
            + 0.35 * _recency_bonus(_year(movie), newest_year)
            + 0.25 * (_popularity(movie) / max_popularity)
        )

        bits = []
        if genre_hits:
            bits.append("matches " + ", ".join(genre_hits[:2]))
        if keyword_hits:
            bits.append("shares " + ", ".join(keyword_hits[:2]))
        reason = "; ".join(bits) if bits else "well-rated crowd-pleaser"
        scored.append(_result(movie, score, reason.capitalize()))

    scored.sort(key=lambda item: (item["score"], item["vote_count"]), reverse=True)
    return scored[:top_n]


def smart_recommend(
    user_genres: List[str],
    user_rated: Dict[int, float],
    movies: Optional[List[dict]] = None,
    top_n: int = DEFAULT_TOP_N,
) -> List[dict]:
    """v2: weighted profile + IDF keywords + Bayesian ratings + diversity."""
    movies = list(movies) if movies is not None else get_movies()
    top_n = max(0, int(top_n))
    if not movies or top_n == 0:
        return []

    context = _catalogue_context(movies)
    profile = build_user_profile(movies, user_genres, user_rated)
    if not profile["genre_weights"] and not profile["keyword_weights"]:
        ordered = _fallback_order(movies, smart=True, context=context)
        return [
            _result(m, _bayesian_rating(m, context["mean_rating"]), "Acclaimed by viewers with enough votes")
            for m in ordered[:top_n]
        ]

    genre_weights: Counter = profile["genre_weights"]
    keyword_weights: Counter = profile["keyword_weights"]
    rated_ids = profile["rated_ids"]
    newest_year = max((_year(m) for m in movies), default=0)
    max_popularity = context["max_popularity"]

    strong_keywords = {
        keyword: weight
        for keyword, weight in keyword_weights.items()
        if weight >= 0.6 or context["idf"].get(keyword, 1.0) >= 2.5
    }

    candidates: List[dict] = []
    for movie in movies:
        if movie.get("id") in rated_ids:
            continue

        genre_hits = [_key(g) for g in _genres(movie) if _key(g) in genre_weights]
        genre_affinity = sum(genre_weights[g] for g in genre_hits)

        keyword_hits = []
        keyword_affinity = 0.0
        for keyword in _keywords(movie):
            key = _key(keyword)
            if key in strong_keywords:
                weight = keyword_weights[key] * context["idf"].get(key, 1.0)
                keyword_affinity += weight
                keyword_hits.append((keyword, weight))
        keyword_affinity = math.sqrt(keyword_affinity) if keyword_affinity else 0.0

        bayes = _bayesian_rating(movie, context["mean_rating"])
        score = (
            2.9 * genre_affinity
            + 1.35 * keyword_affinity
            + 0.9 * bayes
            + 0.45 * _recency_bonus(_year(movie), newest_year)
            + 0.35 * (_popularity(movie) / max_popularity)
            + 0.12 * math.log1p(_votes(movie))
        )

        keyword_hits.sort(key=lambda item: item[1], reverse=True)
        bits = []
        if genre_hits:
            bits.append("strong " + ", ".join(sorted(set(genre_hits))[:2]) + " match")
        if keyword_hits:
            bits.append("themes: " + ", ".join(k for k, _ in keyword_hits[:2]))
        bits.append(f"{_votes(movie):,} votes")
        candidates.append(
            {
                "movie": movie,
                "base_score": score,
                "reason": "; ".join(bits).capitalize(),
                "genres": {_key(g) for g in _genres(movie)},
            }
        )

    candidates.sort(key=lambda item: (item["base_score"], _votes(item["movie"])), reverse=True)

    selected: List[dict] = []
    genre_usage: Counter = Counter()
    while candidates and len(selected) < top_n:
        best_index = max(
            range(len(candidates)),
            key=lambda i: candidates[i]["base_score"] - 0.55 * sum(genre_usage[g] for g in candidates[i]["genres"]),
        )
        best = candidates.pop(best_index)
        adjusted = best["base_score"] - 0.55 * sum(genre_usage[g] for g in best["genres"])
        selected.append(_result(best["movie"], max(0.0, adjusted), best["reason"]))
        genre_usage.update(best["genres"])

    return selected
