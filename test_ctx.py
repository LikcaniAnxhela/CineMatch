import sys
sys.path.insert(0, '.')
from app.main import app, users, get_cached_movies

# Test creating the user_list
user_list = [{"id": u.id, "name": u.name, "favorite_genres": u.favorite_genres, "classified_movie_count": len(u.classified_movies)} for u in users.values()]
movies = get_cached_movies()
print('user_list length:', len(user_list))
print('movies length:', len(movies))
print('First movie:', movies[0] if movies else 'none')
print('user_list[0]:', user_list[0] if user_list else 'none')