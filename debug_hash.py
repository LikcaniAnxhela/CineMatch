import sys
sys.path.insert(0, '.')
from app.main import app, users

# Test if the user dict values are hashable
for uid, u in users.items():
    try:
        hash(u)
        print(f'User {uid} is hashable')
    except TypeError as e:
        print(f'User {uid} is NOT hashable: {e}')
    
    # Check if the dict itself is hashable
    user_dict = {"id": u.id, "name": u.name, "favorite_genres": u.favorite_genres, "classified_movie_count": len(u.classified_movies)}
    try:
        hash(user_dict)
        print(f'User dict is hashable')
    except TypeError as e:
        print(f'User dict is NOT hashable: {e}')
        # Check which value is unhashable
        for k, v in user_dict.items():
            try:
                hash(v)
            except TypeError as e2:
                print(f'  Value for key "{k}" ({v!r}) is NOT hashable: {e2}')