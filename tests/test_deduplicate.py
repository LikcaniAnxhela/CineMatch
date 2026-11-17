import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.deduplicate import (
    catalogue_stats,
    clean_movie,
    classify_movie_genres,
    classify_movie_keywords,
    deduplicate_movies,
    load_and_deduplicate,
    normalize_title,
    save_cleaned_movies,
)


def test_deduplicate_basic():
    movies = [
        {"id": 1, "title": "Movie A", "year": 2020, "genres": ["Action"]},
        {"id": 2, "title": "Movie B", "year": 2021, "genres": ["Drama"]},
        {"id": 1, "title": "Movie A", "year": 2020, "genres": ["Action"]},
    ]
    result = deduplicate_movies(movies)
    assert len(result) == 2
    assert result[0]["id"] == 1


def test_deduplicate_no_duplicates():
    movies = [
        {"id": 1, "title": "Movie A", "year": 2020, "genres": ["Action"]},
        {"id": 2, "title": "Movie B", "year": 2021, "genres": ["Drama"]},
    ]
    assert len(deduplicate_movies(movies)) == 2


def test_deduplicate_all_duplicates():
    movies = [
        {"id": 1, "title": "Movie A", "year": 2020, "genres": ["Action"]},
        {"id": 1, "title": "Movie A", "year": 2020, "genres": ["Action"]},
        {"id": 1, "title": "Movie A", "year": 2020, "genres": ["Action"]},
    ]
    assert len(deduplicate_movies(movies)) == 1


def test_deduplicate_missing_id():
    movies = [
        {"title": "Movie A"},
        {"id": 2, "title": "Movie B", "year": 2021, "genres": ["Drama"]},
    ]
    result = deduplicate_movies(movies)
    assert len(result) == 1
    assert result[0]["id"] == 2


def test_title_year_duplicates_keep_highest_quality():
    movies = [
        {"id": 10, "title": "Interstellar", "year": 2014, "vote_count": 5, "genres": ["Sci-Fi"]},
        {
            "id": 11,
            "title": "  interstellar ",
            "year": 2014,
            "vote_count": 10867,
            "genres": ["Sci-Fi"],
            "keywords": ["space", "time"],
            "overview": "A team travels through a wormhole.",
        },
    ]
    result = deduplicate_movies(movies)
    assert len(result) == 1
    assert result[0]["id"] == 11
    assert result[0]["title"] == "interstellar"


def test_different_years_are_not_duplicates():
    movies = [
        {"id": 1, "title": "Dune", "year": 1984, "genres": ["Sci-Fi"]},
        {"id": 2, "title": "Dune", "year": 2021, "genres": ["Sci-Fi"]},
    ]
    assert len(deduplicate_movies(movies)) == 2


def test_clean_movie_normalises_fields():
    movie = {
        "id": "42",
        "title": "  A   Title  ",
        "year": "1999",
        "runtime": None,
        "genres": ["Action", "action", " Drama ", ""],
        "keywords": ["Space", "SPACE", None, "future"],
        "vote_average": 12,
        "vote_count": "100",
        "popularity": None,
    }
    cleaned = clean_movie(movie)
    assert cleaned["id"] == 42
    assert cleaned["title"] == "A Title"
    assert cleaned["year"] == 1999
    assert cleaned["runtime"] is None
    assert cleaned["genres"] == ["Action", "Drama"]
    assert cleaned["keywords"] == ["space", "future"]
    assert cleaned["vote_average"] == 10.0
    assert cleaned["vote_count"] == 100
    assert cleaned["popularity"] == 0.0


def test_clean_movie_rejects_invalid_records():
    assert clean_movie(None) is None
    assert clean_movie({"title": "No id"}) is None
    assert clean_movie({"id": 1, "title": "   "}) is None
    assert clean_movie({"id": "not-an-int", "title": "Bad id"}) is None


def test_normalize_title():
    assert normalize_title("Amélie") == "amelie"
    assert normalize_title("Spider-Man: No Way Home") == "spider man no way home"


def test_classify_movie_genres():
    movies = [
        {"id": 1, "title": "Movie A", "genres": ["Action", "Adventure"]},
        {"id": 2, "title": "Movie B", "genres": ["Action", "Drama"]},
        {"id": 3, "title": "Movie C", "genres": ["Comedy"]},
    ]
    result = classify_movie_genres(movies)
    assert result["action"] == [1, 2]
    assert 3 in result["comedy"]


def test_classify_movie_keywords():
    movies = [
        {"id": 1, "title": "Movie A", "keywords": ["space", "future"]},
        {"id": 2, "title": "Movie B", "keywords": ["space", "robot"]},
    ]
    result = classify_movie_keywords(movies)
    assert result["space"] == [1, 2]
    assert result["robot"] == [2]


def test_load_and_deduplicate_uses_custom_path(tmp_path):
    source = tmp_path / "movies.json"
    test_movies = [
        {"id": 1, "title": "Movie A", "genres": ["Action"], "year": 2020},
        {"id": 2, "title": "Movie B", "genres": ["Comedy"], "year": 2021},
        {"id": 1, "title": "Movie A", "genres": ["Action"], "year": 2020},
    ]
    source.write_text(json.dumps(test_movies), encoding="utf-8")

    result = load_and_deduplicate(source)
    assert len(result) == 2


def test_save_cleaned_movies_never_touches_source(tmp_path):
    source = tmp_path / "movies.json"
    target = tmp_path / "movies_clean.json"
    original = [{"id": 1, "title": "Movie A", "year": 2020, "genres": ["Action"]}]
    source.write_text(json.dumps(original), encoding="utf-8")

    saved_path = save_cleaned_movies(path=target, source_path=source)

    assert saved_path == str(target)
    assert json.loads(source.read_text(encoding="utf-8")) == original
    assert len(json.loads(target.read_text(encoding="utf-8"))) == 1


def test_catalogue_stats():
    movies = [
        {"id": 1, "title": "A", "genres": ["Action"], "year": 2000},
        {"id": 2, "title": "B", "genres": ["Action", "Drama"], "year": None},
    ]
    stats = catalogue_stats(movies)
    assert stats["movie_count"] == 2
    assert stats["missing_year"] == 1
    assert stats["year_min"] == 2000
    assert ("Action", 2) in stats["top_genres"]


def test_real_catalogue_invariants():
    from app.deduplicate import DATA_DIR

    raw_path = DATA_DIR / "movies.json"
    if not raw_path.exists():
        return

    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    cleaned = load_and_deduplicate(raw_path)

    assert len(raw) == 4825
    assert len(cleaned) == 4822
    ids = [movie["id"] for movie in cleaned]
    assert len(ids) == len(set(ids))
    keys = [(normalize_title(m["title"]), m.get("year")) for m in cleaned]
    assert len(keys) == len(set(keys))
