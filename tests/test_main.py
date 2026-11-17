import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastapi.testclient import TestClient
from app.main import app, users, next_user_id, load_movies

client = TestClient(app)


def _reset_state():
    """Reset module-level state for test isolation."""
    import app.main
    app.main.users = {}
    app.main.next_user_id = 1


def test_create_user():
    _reset_state()
    response = client.post("/users/", json={"name": "Alice"})
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Alice"
    assert data["id"] == 1


def test_create_multiple_users():
    _reset_state()
    response = client.post("/users/", json={"name": "Alice"})
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Alice"
    assert data["id"] == 1

    response = client.post("/users/", json={"name": "Bob"})
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Bob"
    assert data["id"] == 2


def test_list_users():
    _reset_state()
    client.post("/users/", json={"name": "Alice"})
    client.post("/users/", json={"name": "Bob"})
    response = client.get("/users/")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2


def test_get_user():
    _reset_state()
    client.post("/users/", json={"name": "Alice"})
    user_id = 1
    response = client.get(f"/users/{user_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Alice"


def test_get_user_not_found():
    _reset_state()
    response = client.get("/users/999")
    assert response.status_code == 404


def test_add_user_genres():
    _reset_state()
    client.post("/users/", json={"name": "Alice"})
    response = client.post("/users/1/genres", json=["Action", "Adventure"])
    assert response.status_code == 200
    data = response.json()
    assert set(data["favorite_genres"]) == {"Action", "Adventure"}


def test_classify_movie():
    _reset_state()
    client.post("/users/", json={"name": "Alice"})
    movies = load_movies()
    if movies:
        movie_id = movies[0]["id"]
        response = client.post(f"/users/1/classify/{movie_id}", json={"rating": 4.5})
        assert response.status_code == 200
        data = response.json()
        assert data["classified_movie_count"] == 1


def test_classify_movie_invalid_rating():
    _reset_state()
    client.post("/users/", json={"name": "Alice"})
    movies = load_movies()
    if movies:
        movie_id = movies[0]["id"]
        response = client.post(f"/users/1/classify/{movie_id}", json={"rating": 10})
        assert response.status_code == 400


def test_recommend_naive_mode():
    _reset_state()
    client.post("/users/", json={"name": "Alice"})
    client.post("/users/1/genres", json=["Action"])

    movies = load_movies()
    if movies:
        movie_id = movies[0]["id"]
        client.post(f"/users/1/classify/{movie_id}", json={"rating": 5.0})

    response = client.get("/recommend/1?mode=naive")
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "naive"
    assert len(data["recommended"]) > 0


def test_recommend_smart_mode():
    _reset_state()
    client.post("/users/", json={"name": "Alice"})
    client.post("/users/1/genres", json=["Action"])

    movies = load_movies()
    if movies:
        movie_id = movies[0]["id"]
        client.post(f"/users/1/classify/{movie_id}", json={"rating": 5.0})

    response = client.get("/recommend/1?mode=smart")
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "smart"
    assert len(data["recommended"]) > 0


def test_compare_modes():
    _reset_state()
    client.post("/users/", json={"name": "Alice"})
    client.post("/users/1/genres", json=["Action"])

    movies = load_movies()
    if movies:
        movie_id = movies[0]["id"]
        client.post(f"/users/1/classify/{movie_id}", json={"rating": 5.0})

    response = client.get("/compare/1")
    assert response.status_code == 200
    data = response.json()
    assert "mode_naive" in data
    assert "mode_smart" in data
    assert data["favorite_genres"] == ["Action"]
    assert data["overlap"] >= 0


def test_homepage_renders():
    _reset_state()
    response = client.get("/")
    assert response.status_code == 200
    assert "CineMatch" in response.text
    assert 'id="dashboard"' in response.text
    assert 'id="movie-grid"' in response.text
    assert "Browse the collection" in response.text


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["movies"] > 4000


def test_movies_api_search_filter_and_pagination():
    response = client.get("/api/movies", params={"search": "star", "genre": "Action", "page_size": 5})
    assert response.status_code == 200
    data = response.json()
    assert data["total"] > 0
    assert data["page_size"] == 5
    assert len(data["movies"]) <= 5
    for movie in data["movies"]:
        haystack = " ".join(
            [
                movie.get("title", ""),
                movie.get("overview", ""),
                " ".join(movie.get("keywords", [])),
            ]
        ).casefold()
        assert "star" in haystack

    response = client.get("/api/movies", params={"sort": "newest", "page_size": 3})
    newest = response.json()["movies"]
    years = [m["year"] for m in newest if isinstance(m["year"], int)]
    assert years == sorted(years, reverse=True)


def test_ui_dashboard_partial():
    _reset_state()
    client.post("/users/", json={"name": "Dash"})
    response = client.get("/ui/dashboard?user_id=1")
    assert response.status_code == 200
    assert 'id="dashboard"' in response.text
    assert "Naive" in response.text and "Smart" in response.text


def test_ui_create_user_form_workflow():
    _reset_state()
    response = client.post("/ui/users", data={"name": "Nova"})
    assert response.status_code == 200
    assert "Nova" in response.text
    assert 'hx-swap-oob="true"' in response.text

    response = client.post("/ui/genres", data={"user_id": "1", "genre": "Documentary"})
    assert response.status_code == 200
    assert "Documentary" in response.text

    movies = load_movies()
    movie_id = movies[0]["id"]
    response = client.post(
        "/ui/ratings",
        data={"user_id": "1", "movie_id": str(movie_id), "rating": "4"},
    )
    assert response.status_code == 200
    assert "Rated" in response.text


def test_ui_rating_validation_message():
    _reset_state()
    client.post("/users/", json={"name": "Dash"})
    response = client.post(
        "/ui/ratings",
        data={"user_id": "1", "movie_id": "19995", "rating": "9"},
    )
    assert response.status_code == 200
    assert "between 1 and 5" in response.text


def test_missing_user_and_movie_errors():
    _reset_state()
    assert client.get("/users/99").status_code == 404
    assert client.get("/recommend/99").status_code == 404
    client.post("/users/", json={"name": "Alice"})
    assert client.post("/users/1/classify/999999999", json={"rating": 4}).status_code == 404