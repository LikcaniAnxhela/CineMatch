"""CineMatch FastAPI application.

A fully offline movie recommendation experience built with FastAPI, Jinja2,
Tailwind CSS (CDN) and HTMX.  The raw catalogue is never written to; cleaned
records are cached in memory and can be persisted to ``data/movies_clean.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.deduplicate import (
    CLEAN_PATH,
    SOURCE_PATH,
    catalogue_stats,
    load_and_deduplicate,
    load_movies as load_movies_file,
    save_cleaned_movies,
)
from app.recommender import naive_recommend, smart_recommend

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TEMPLATES_DIR = PROJECT_ROOT / "templates"
STATIC_DIR = TEMPLATES_DIR / "static"

app = FastAPI(title="CineMatch", description="Offline movie recommendation studio")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ---------------------------------------------------------------------------
# Models & in-memory user store
# ---------------------------------------------------------------------------


class UserBase(BaseModel):
    id: int
    name: str
    favorite_genres: List[str] = Field(default_factory=list)
    classified_movies: Dict[int, float] = Field(default_factory=dict)


class UserCreate(BaseModel):
    name: str = Field(..., min_length=1, description="User name")


class UserPublic(BaseModel):
    id: int
    name: str
    favorite_genres: List[str]
    classified_movie_count: int


users: Dict[int, UserBase] = {}
next_user_id: int = 1


def _seed_demo_users() -> None:
    """Seed two friendly demo profiles so the homepage is useful immediately."""
    global next_user_id
    demo_profiles = [
        {
            "name": "Alex Rivera",
            "favorite_genres": ["Action", "Science Fiction", "Thriller"],
            "classified_movies": {19995: 5.0, 157336: 5.0, 155: 4.5, 27205: 4.5},
        },
        {
            "name": "Maya Chen",
            "favorite_genres": ["Drama", "Romance", "Comedy"],
            "classified_movies": {597: 4.5, 278: 5.0, 680: 4.0, 900012: 4.5},
        },
    ]
    for profile in demo_profiles:
        user = UserBase(id=next_user_id, **profile)
        users[user.id] = user
        next_user_id += 1


_seed_demo_users()


# ---------------------------------------------------------------------------
# Movie cache & helpers
# ---------------------------------------------------------------------------


movies_cache: Optional[List[dict]] = None
stats_cache: Optional[dict] = None


def load_movies() -> List[dict]:
    """Public helper kept for backwards compatibility and tests."""
    return load_movies_file(SOURCE_PATH)


def get_cached_movies() -> List[dict]:
    global movies_cache
    if movies_cache is None:
        if CLEAN_PATH.exists():
            try:
                with open(CLEAN_PATH, "r", encoding="utf-8-sig") as handle:
                    data = json.load(handle)
                if isinstance(data, list) and data:
                    movies_cache = data
            except (OSError, json.JSONDecodeError):
                movies_cache = None
        if not movies_cache:
            movies_cache = load_and_deduplicate(SOURCE_PATH)
    return movies_cache


def get_stats() -> dict:
    global stats_cache
    if stats_cache is None:
        stats_cache = catalogue_stats(get_cached_movies())
    return stats_cache


def _invalidate_cache() -> None:
    global movies_cache, stats_cache
    movies_cache = None
    stats_cache = None


def movie_index() -> Dict[int, dict]:
    return {movie.get("id"): movie for movie in get_cached_movies()}


def _movie_view(movie: dict) -> dict:
    view = dict(movie)
    movie_id = movie.get("id") or 0
    title = movie.get("title") or "Untitled"
    view["hue"] = int(movie_id) % 360
    view["initial"] = title.strip()[:1].upper() or "?"
    view["year"] = movie.get("year") or "—"
    view["rating"] = round(float(movie.get("vote_average") or 0), 1)
    view["genres"] = movie.get("genres") or []
    return view


def _user_public(user: UserBase) -> UserPublic:
    return UserPublic(
        id=user.id,
        name=user.name,
        favorite_genres=list(user.favorite_genres),
        classified_movie_count=len(user.classified_movies),
    )


def _user_list() -> List[dict]:
    return [_user_public(user).model_dump() for user in users.values()]


def _dedupe_ordered(values: List[str]) -> List[str]:
    result: List[str] = []
    seen: set = set()
    for value in values:
        text = str(value).strip()
        if not text or text.casefold() in seen:
            continue
        seen.add(text.casefold())
        result.append(text)
    return result


def _all_genres() -> List[dict]:
    counts: Dict[str, int] = {}
    display: Dict[str, str] = {}
    for movie in get_cached_movies():
        for genre in movie.get("genres") or []:
            key = str(genre).casefold()
            display.setdefault(key, str(genre))
            counts[key] = counts.get(key, 0) + 1
    return [
        {"name": display[key], "count": count}
        for key, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)
    ]


def _resolve_user(user_id: Optional[int]) -> Optional[UserBase]:
    if user_id is not None:
        return users.get(user_id)
    if users:
        return next(iter(users.values()))
    return None


async def _read_payload(request: Request) -> Dict[str, Any]:
    """Parse JSON or form-urlencoded bodies without python-multipart."""
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            data = await request.json()
            return data if isinstance(data, dict) else {}
        except (ValueError, json.JSONDecodeError):
            return {}
    raw = (await request.body()).decode("utf-8", errors="replace")
    parsed = parse_qs(raw, keep_blank_values=True)
    return {key: values[-1] if values else "" for key, values in parsed.items()}


def _dashboard_context(
    user_id: Optional[int],
    *,
    toast: str = "",
    toast_kind: str = "success",
    oob: bool = False,
) -> dict:
    movie_map = movie_index()
    selected = _resolve_user(user_id)

    context: dict = {
        "user": selected,
        "selected_user_id": selected.id if selected else 0,
        "user_list": _user_list(),
        "toast": toast,
        "toast_kind": toast_kind,
        "oob": oob,
        "catalogue": get_stats(),
    }

    if selected is None:
        context.update(
            {
                "favorite_genres": [],
                "rated_movies": [],
                "naive": [],
                "smart": [],
                "comparison_rows": [],
                "overlap": 0,
                "rating_count": 0,
            }
        )
        return context

    naive = naive_recommend(selected.favorite_genres, selected.classified_movies, top_n=8)
    smart = smart_recommend(selected.favorite_genres, selected.classified_movies, top_n=8)

    rated_movies = []
    for movie_id, rating in selected.classified_movies.items():
        movie = movie_map.get(movie_id)
        if movie is None:
            continue
        entry = _movie_view(movie)
        entry["user_rating"] = rating
        rated_movies.append(entry)
    rated_movies.sort(key=lambda item: item["user_rating"], reverse=True)

    overlap = len({item["id"] for item in naive} & {item["id"] for item in smart})
    comparison_rows = []
    for index in range(max(len(naive), len(smart))):
        comparison_rows.append(
            {
                "rank": index + 1,
                "naive": naive[index] if index < len(naive) else None,
                "smart": smart[index] if index < len(smart) else None,
            }
        )

    context.update(
        {
            "favorite_genres": selected.favorite_genres,
            "rated_movies": rated_movies,
            "naive": naive,
            "smart": smart,
            "comparison_rows": comparison_rows,
            "overlap": overlap,
            "rating_count": len(selected.classified_movies),
        }
    )
    return context


def _render_dashboard(request: Request, context: dict, status_code: int = 200) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "partials/dashboard.html",
        context,
        status_code=status_code,
    )


def _filter_movies(
    movies: List[dict],
    search: str = "",
    genre: str = "",
    sort: str = "popularity",
) -> List[dict]:
    query = (search or "").strip().casefold()
    genre_key = (genre or "").strip().casefold()
    results = list(movies)

    if query:
        tokens = [token for token in query.split() if token]

        def matches(movie: dict) -> bool:
            haystack = " ".join(
                [
                    str(movie.get("title") or ""),
                    str(movie.get("original_title") or ""),
                    str(movie.get("overview") or ""),
                    " ".join(str(g) for g in (movie.get("genres") or [])),
                    " ".join(str(k) for k in (movie.get("keywords") or [])),
                ]
            ).casefold()
            return all(token in haystack for token in tokens)

        results = [movie for movie in results if matches(movie)]

    if genre_key and genre_key != "all":
        results = [
            movie
            for movie in results
            if genre_key in {str(g).casefold() for g in (movie.get("genres") or [])}
        ]

    sorters = {
        "popularity": lambda m: (
            float(m.get("popularity") or 0),
            float(m.get("vote_average") or 0),
        ),
        "rating": lambda m: (
            float(m.get("vote_average") or 0),
            int(m.get("vote_count") or 0),
        ),
        "newest": lambda m: (int(m.get("year") or 0), float(m.get("vote_average") or 0)),
        "oldest": lambda m: (-int(m.get("year") or 9999), float(m.get("vote_average") or 0)),
        "title": lambda m: str(m.get("title") or "").casefold(),
    }
    sorter = sorters.get(sort, sorters["popularity"])
    reverse = sort != "title" and sort != "oldest"
    results.sort(key=sorter, reverse=reverse)
    return results


def _grid_context(
    *,
    search: str,
    genre: str,
    sort: str,
    page: int,
    page_size: int = 18,
) -> dict:
    filtered = _filter_movies(get_cached_movies(), search, genre, sort)
    total = len(filtered)
    pages = max(1, (total + page_size - 1) // page_size)
    page = max(1, min(page, pages))
    start = (page - 1) * page_size
    page_movies = [_movie_view(movie) for movie in filtered[start : start + page_size]]
    return {
        "movies": page_movies,
        "total": total,
        "page": page,
        "pages": pages,
        "page_size": page_size,
        "search": search,
        "genre": genre,
        "sort": sort,
        "genres": _all_genres(),
    }


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def root(request: Request, user_id: Optional[int] = None):
    if user_id is None:
        selected = _resolve_user(None)
        dashboard_user_id = selected.id if selected else 0
    else:
        dashboard_user_id = user_id
    dashboard = _dashboard_context(dashboard_user_id)
    grid = _grid_context(search="", genre="", sort="popularity", page=1)
    featured = [
        _movie_view(movie)
        for movie in sorted(
            [m for m in get_cached_movies() if int(m.get("vote_count") or 0) >= 500],
            key=lambda m: (float(m.get("vote_average") or 0), int(m.get("vote_count") or 0)),
            reverse=True,
        )[:3]
    ]
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "title": "CineMatch",
            "grid": grid,
            "featured": featured,
            **dashboard,
        },
    )


@app.get("/ui/dashboard", response_class=HTMLResponse)
async def ui_dashboard(request: Request, user_id: Optional[int] = None):
    context = _dashboard_context(user_id, oob=True)
    return _render_dashboard(request, context)


@app.get("/ui/movies", response_class=HTMLResponse)
async def ui_movies(
    request: Request,
    search: str = "",
    genre: str = "",
    sort: str = "popularity",
    page: int = 1,
):
    context = _grid_context(search=search, genre=genre, sort=sort, page=page)
    return templates.TemplateResponse(request, "partials/movie_grid.html", context)


@app.post("/ui/users", response_class=HTMLResponse)
async def ui_create_user(request: Request):
    payload = await _read_payload(request)
    name = str(payload.get("name", "")).strip()
    if not name:
        context = _dashboard_context(0, toast="Please enter a name.", toast_kind="error", oob=True)
        return _render_dashboard(request, context, status_code=200)
    if len(name) > 60:
        context = _dashboard_context(0, toast="Name must be 60 characters or fewer.", toast_kind="error", oob=True)
        return _render_dashboard(request, context, status_code=200)

    global next_user_id
    user = UserBase(id=next_user_id, name=name)
    users[user.id] = user
    next_user_id += 1

    context = _dashboard_context(user.id, toast=f"Welcome, {name}!", oob=True)
    return _render_dashboard(request, context)


@app.post("/ui/genres", response_class=HTMLResponse)
async def ui_add_genre(request: Request):
    payload = await _read_payload(request)
    user = users.get(_to_int(payload.get("user_id")))
    genre = str(payload.get("genre", "")).strip()
    if user is None:
        context = _dashboard_context(0, toast="Select or create a user first.", toast_kind="error", oob=True)
        return _render_dashboard(request, context)
    if not genre:
        context = _dashboard_context(user.id, toast="Enter a genre to add.", toast_kind="error", oob=True)
        return _render_dashboard(request, context)

    user.favorite_genres = _dedupe_ordered([*user.favorite_genres, genre])
    context = _dashboard_context(user.id, toast=f"Added {genre} to your taste profile.", oob=True)
    return _render_dashboard(request, context)


@app.post("/ui/genres/remove", response_class=HTMLResponse)
async def ui_remove_genre(request: Request):
    payload = await _read_payload(request)
    user = users.get(_to_int(payload.get("user_id")))
    genre = str(payload.get("genre", "")).strip().casefold()
    if user is None:
        context = _dashboard_context(0, toast="Select a user first.", toast_kind="error", oob=True)
        return _render_dashboard(request, context)

    user.favorite_genres = [g for g in user.favorite_genres if g.casefold() != genre]
    context = _dashboard_context(user.id, toast="Genre removed.", oob=True)
    return _render_dashboard(request, context)


@app.post("/ui/ratings", response_class=HTMLResponse)
async def ui_rate_movie(request: Request):
    payload = await _read_payload(request)
    user = users.get(_to_int(payload.get("user_id")))
    movie_id = _to_int(payload.get("movie_id"))
    rating = _to_float(payload.get("rating"))

    if user is None:
        context = _dashboard_context(0, toast="Select or create a user before rating.", toast_kind="error", oob=True)
        return _render_dashboard(request, context)
    if movie_id is None or movie_id not in movie_index():
        context = _dashboard_context(user.id, toast="That movie could not be found.", toast_kind="error", oob=True)
        return _render_dashboard(request, context)
    if rating is None or rating < 1 or rating > 5:
        context = _dashboard_context(user.id, toast="Ratings must be between 1 and 5.", toast_kind="error", oob=True)
        return _render_dashboard(request, context)

    user.classified_movies[movie_id] = rating
    movie = movie_index()[movie_id]
    context = _dashboard_context(
        user.id,
        toast=f"Rated {movie.get('title', 'movie')} {rating:g}/5.",
        oob=True,
    )
    return _render_dashboard(request, context)


@app.post("/ui/ratings/remove", response_class=HTMLResponse)
async def ui_remove_rating(request: Request):
    payload = await _read_payload(request)
    user = users.get(_to_int(payload.get("user_id")))
    movie_id = _to_int(payload.get("movie_id"))
    if user is None:
        context = _dashboard_context(0, toast="Select a user first.", toast_kind="error", oob=True)
        return _render_dashboard(request, context)
    if movie_id is not None:
        user.classified_movies.pop(movie_id, None)
    context = _dashboard_context(user.id, toast="Rating removed.", oob=True)
    return _render_dashboard(request, context)


def _to_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# JSON API - users
# ---------------------------------------------------------------------------


@app.post("/users/", response_model=UserPublic)
async def create_user(user_create: UserCreate) -> UserPublic:
    global next_user_id
    name = user_create.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name cannot be empty")
    user = UserBase(id=next_user_id, name=name)
    users[user.id] = user
    next_user_id += 1
    return _user_public(user)


@app.get("/users/", response_model=List[UserPublic])
async def list_users() -> List[UserPublic]:
    return [_user_public(user) for user in users.values()]


@app.get("/users/{user_id}", response_model=UserPublic)
async def get_user(user_id: int) -> UserPublic:
    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
    return _user_public(users[user_id])


@app.post("/users/{user_id}/genres", response_model=UserPublic)
async def add_user_genres(user_id: int, genres: List[str]) -> UserPublic:
    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
    users[user_id].favorite_genres = _dedupe_ordered(genres)
    return _user_public(users[user_id])


@app.delete("/users/{user_id}/genres/{genre}", response_model=UserPublic)
async def delete_user_genre(user_id: int, genre: str) -> UserPublic:
    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
    key = genre.casefold()
    users[user_id].favorite_genres = [g for g in users[user_id].favorite_genres if g.casefold() != key]
    return _user_public(users[user_id])


@app.post("/users/{user_id}/classify/{movie_id}", response_model=UserPublic)
async def classify_movie(user_id: int, movie_id: int, request: Request) -> UserPublic:
    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
    if movie_id not in movie_index():
        raise HTTPException(status_code=404, detail="Movie not found")
    payload = await _read_payload(request)
    rating = _to_float(payload.get("rating", 5.0))
    if rating is None or rating < 1 or rating > 5:
        raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")
    users[user_id].classified_movies[movie_id] = rating
    return _user_public(users[user_id])


@app.delete("/users/{user_id}/classify/{movie_id}", response_model=UserPublic)
async def delete_movie_rating(user_id: int, movie_id: int) -> UserPublic:
    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
    users[user_id].classified_movies.pop(movie_id, None)
    return _user_public(users[user_id])


@app.get("/users/{user_id}/ratings")
async def user_ratings(user_id: int) -> dict:
    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
    user = users[user_id]
    movie_map = movie_index()
    ratings = []
    for movie_id, rating in user.classified_movies.items():
        movie = movie_map.get(movie_id)
        if movie:
            ratings.append(
                {
                    "movie_id": movie_id,
                    "title": movie.get("title"),
                    "year": movie.get("year"),
                    "rating": rating,
                }
            )
    return {"user_id": user_id, "ratings": ratings}


# ---------------------------------------------------------------------------
# JSON API - catalogue & recommendations
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "movies": len(get_cached_movies()), "users": len(users)}


@app.get("/api/movies")
async def api_movies(
    search: str = "",
    genre: str = "",
    sort: str = "popularity",
    page: int = 1,
    page_size: int = 18,
) -> dict:
    page_size = max(1, min(60, int(page_size)))
    grid = _grid_context(search=search, genre=genre, sort=sort, page=page, page_size=page_size)
    return {
        "movies": grid["movies"],
        "total": grid["total"],
        "page": grid["page"],
        "pages": grid["pages"],
        "page_size": grid["page_size"],
        "search": search,
        "genre": genre,
        "sort": sort,
    }


@app.get("/api/genres")
async def api_genres() -> dict:
    return {"genres": _all_genres()}


@app.get("/recommend/{user_id}")
async def recommend(
    user_id: int,
    mode: str = "naive",
    top_n: int = 12,
) -> dict:
    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
    if mode not in ("naive", "smart"):
        raise HTTPException(status_code=400, detail="Mode must be 'naive' or 'smart'")
    top_n = max(1, min(50, int(top_n)))

    user = users[user_id]
    movies = get_cached_movies()
    if mode == "naive":
        recommended = naive_recommend(user.favorite_genres, user.classified_movies, movies, top_n=top_n)
    else:
        recommended = smart_recommend(user.favorite_genres, user.classified_movies, movies, top_n=top_n)

    return {
        "user_id": user_id,
        "mode": mode,
        "favorite_genres": user.favorite_genres,
        "recommended": recommended,
        "total_movies": len(movies),
    }


@app.get("/compare/{user_id}")
async def compare(user_id: int, top_n: int = 10) -> dict:
    if user_id not in users:
        raise HTTPException(status_code=404, detail="User not found")
    top_n = max(1, min(50, int(top_n)))
    user = users[user_id]
    movies = get_cached_movies()
    naive_rec = naive_recommend(user.favorite_genres, user.classified_movies, movies, top_n=top_n)
    smart_rec = smart_recommend(user.favorite_genres, user.classified_movies, movies, top_n=top_n)
    overlap = len({item["id"] for item in naive_rec} & {item["id"] for item in smart_rec})
    return {
        "user_id": user_id,
        "mode_naive": {"title": "Naive Recommendations (v1)", "movies": naive_rec},
        "mode_smart": {"title": "Smart Recommendations (v2)", "movies": smart_rec},
        "favorite_genres": user.favorite_genres,
        "overlap": overlap,
    }


@app.post("/data/deduplicate")
async def deduplicate_data() -> dict:
    if not SOURCE_PATH.exists():
        raise HTTPException(status_code=404, detail="movies.json not found")
    cleaned = load_and_deduplicate(SOURCE_PATH)
    save_cleaned_movies(cleaned, CLEAN_PATH)
    global movies_cache, stats_cache
    movies_cache = cleaned
    stats_cache = catalogue_stats(cleaned)
    return {
        "message": "Deduplication complete",
        "movie_count": len(cleaned),
        "saved_to": str(CLEAN_PATH),
        "stats": stats_cache,
    }
