import sys
sys.path.insert(0, '.')
from app.main import app, get_cached_movies

movies = get_cached_movies()
print(f'Number of movies: {len(movies)}')
if movies:
    print(f'First movie: {movies[0]}')
    print(f'First movie type: {type(movies[0])}')
    # Check if any movie has unhashable values
    for i, m in enumerate(movies[:5]):
        try:
            hash(m)
            print(f'Movie {i} is hashable')
        except TypeError as e:
            print(f'Movie {i} is NOT hashable: {e}')
            # Check what's unhashable
            for k, v in m.items():
                try:
                    hash(v)
                except TypeError as e2:
                    print(f'  Key "{k}": {v!r} is unhashable: {e2}')