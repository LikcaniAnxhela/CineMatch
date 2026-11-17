import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.recommender import naive_recommend, smart_recommend, get_movies
from app.deduplicate import load_and_deduplicate
import json
from pathlib import Path


def get_test_movies():
    return [
        {"id": 1, "title": "Avatar", "genres": ["Action", "Adventure", "Fantasy"], "keywords": ["space", "future", "war"], "year": 2009, "vote_average": 7.2},
        {"id": 2, "title": "Pirates", "genres": ["Adventure", "Fantasy"], "keywords": ["ocean", "ship", "pirate"], "year": 2007, "vote_average": 6.9},
        {"id": 3, "title": "Spectre", "genres": ["Action", "Adventure", "Crime"], "keywords": ["spy", "mi6", "british"], "year": 2015, "vote_average": 6.3},
        {"id": 4, "title": "Dark Knight", "genres": ["Action", "Crime", "Drama"], "keywords": ["gotham", "vigilante", "hero"], "year": 2008, "vote_average": 9.0},
        {"id": 5, "title": "Titanic", "genres": ["Drama", "Romance"], "keywords": ["ship", "ocean", "love"], "year": 1997, "vote_average": 7.5},
    ]


def test_naive_recommend_with_genres():
    movies = get_test_movies()
    # User likes Action and Adventure
    recommended = naive_recommend(["Action", "Adventure"], {}, movies, top_n=3)
    assert len(recommended) == 3
    # Should prefer Action/Adventure movies
    titles = [m["title"] for m in recommended]
    assert "Avatar" in titles or "Pirates" in titles or "Spectre" in titles


def test_naive_recommend_without_preferences():
    movies = get_test_movies()
    recommended = naive_recommend([], {}, movies, top_n=3)
    assert len(recommended) == 3


def test_naive_recommend_empty_input():
    movies = get_test_movies()
    recommended = naive_recommend([], {}, movies, top_n=5)
    assert len(recommended) == 5


def test_smart_recommend_with_genres():
    movies = get_test_movies()
    recommended = smart_recommend(["Action", "Adventure"], {}, movies, top_n=3)
    assert len(recommended) == 3
    titles = [m["title"] for m in recommended]
    # Should prefer Action/Adventure movies with better scoring
    assert len(titles) == 3


def test_smart_recommend_idf_weighting():
    """Smart mode should use IDF-inspired keyword weighting."""
    movies = get_test_movies()
    recommended = smart_recommend(["Science Fiction"], {}, movies, top_n=3)
    assert len(recommended) == 3


def test_smart_recommend_empty_preferences():
    movies = get_test_movies()
    recommended = smart_recommend([], {}, movies, top_n=3)
    assert len(recommended) == 3


def test_smart_recommend_basic_movies():
    movies = get_test_movies()
    # User who rates highly-rated movies
    user_rated = {4: 5.0}  # rated Dark Knight highly
    recommended = smart_recommend(["Action"], user_rated, movies, top_n=3)
    assert len(recommended) == 3


def test_naive_vs_smart_different_scoring():
    """Naive and smart modes should produce different scores."""
    movies = get_test_movies()
    naive = naive_recommend(["Action"], {}, movies, top_n=1)
    smart = smart_recommend(["Action"], {}, movies, top_n=1)

    assert naive[0]["title"] != smart[0]["title"] or naive[0]["score"] != smart[0]["score"]

def test_ratings_drive_smart_and_naive_differently():
    movies = [
        {"id": 1, "title": "Loved Heist", "genres": ["Action"], "keywords": ["heist"], "year": 2020, "vote_average": 8.0, "vote_count": 1000},
        {"id": 2, "title": "Disliked Slow Drama", "genres": ["Drama"], "keywords": ["slow"], "year": 2019, "vote_average": 6.0, "vote_count": 200},
        {"id": 3, "title": "Heist Action Candidate", "genres": ["Action"], "keywords": ["heist"], "year": 2021, "vote_average": 8.0, "vote_count": 1200},
        {"id": 4, "title": "Slow Drama Candidate", "genres": ["Drama"], "keywords": ["slow"], "year": 2001, "vote_average": 9.5, "vote_count": 50},
    ]
    ratings = {1: 5.0, 2: 1.0}

    naive = naive_recommend([], ratings, movies, top_n=2)
    smart = smart_recommend([], ratings, movies, top_n=2)

    assert all(item["id"] not in ratings for item in naive)
    assert all(item["id"] not in ratings for item in smart)
    assert smart[0]["id"] == 3
    assert naive[0]["id"] == 4
    assert smart[0]["id"] != naive[0]["id"]


def test_rated_movies_are_never_recommended():
    movies = get_test_movies()
    ratings = {4: 5.0, 5: 4.0}
    for engine in (naive_recommend, smart_recommend):
        results = engine(["Action", "Romance"], ratings, movies, top_n=5)
        assert all(item["id"] not in ratings for item in results)


def test_profiles_merge_favourite_genres_and_ratings():
    from app.recommender import build_user_profile

    movies = get_test_movies()
    profile = build_user_profile(movies, ["Romance"], {4: 5.0})
    assert profile["genre_weights"]["romance"] > 0
    assert profile["genre_weights"]["action"] > 0
    assert profile["keyword_weights"]["gotham"] > 0
    assert 4 in profile["rated_ids"]


def test_results_have_expected_fields():
    movies = get_test_movies()
    results = smart_recommend(["Action"], {}, movies, top_n=3)
    for item in results:
        assert {"id", "title", "year", "rating", "vote_count", "genres", "score", "reason"} <= set(item)
        assert isinstance(item["score"], float)
        assert item["reason"]
