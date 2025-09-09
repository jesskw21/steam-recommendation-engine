import pandas as pd
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import argparse

# Controls how much gets printed to the notebook/console
VERBOSE_OUTPUT = True

print("Steam Game recommendation")

# Names of the files to be loaded
games_file = "games.csv"
users_file = "users.csv"
recommendations_file = "recommendations.csv"
metadata_file = "games_metadata.json"

# Check if files exist before loading them
missing_files = []
for file in [games_file, users_file, recommendations_file, metadata_file]:
    if not os.path.exists(file):
        missing_files.append(file)

try:
    games_df = pd.read_csv(games_file)
    users_df = pd.read_csv(users_file)
    recommendations_df = pd.read_csv(recommendations_file, nrows=1000)

    metadata_list = []
    with open(metadata_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                metadata_list.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    metadata_df = pd.DataFrame(metadata_list)

    print("Inspecting the Games DataFrame")
    if VERBOSE_OUTPUT:
        print("First 5 rows:")
        print(games_df.head())
        print("\nData types and non-null counts:")
        games_df.info()
        print("Statistical summary:")
        print(games_df.describe())
    print(f"\nShape: {games_df.shape}")

    print("\n\n Inspecting the Users DataFrame")
    if VERBOSE_OUTPUT:
        print("First 5 rows:")
        print(users_df.head())
        print("\nData types and non-null counts:")
        users_df.info()
    print(f"\nShape: {users_df.shape}")

    print("\n\n Inspecting the Recommendations DataFrame")
    if VERBOSE_OUTPUT:
        print("First 5 rows:")
        print(recommendations_df.head())
        print("\nData types and non-null counts:")
        recommendations_df.info()
    print(f"\nShape: {recommendations_df.shape}")

    print("\n\n Inspecting the Metadata DataFrame")
    if VERBOSE_OUTPUT:
        print("First 5 rows:")
        print(metadata_df.head())
        print("\nData types and non-null counts:")
        metadata_df.info()
    print(f"\nShape: {metadata_df.shape}")

    print("\nNext Steps for Recommendation Engine")
    
    print("\n1. EXPLORING DATA RELATIONSHIPS:")
    print("Columns in each dataset:")
    print(f"Games: {list(games_df.columns)}")
    print(f"Users: {list(users_df.columns)}")
    print(f"Recommendations: {list(recommendations_df.columns)}")
    print(f"Metadata: {list(metadata_df.columns)}")

    print("\n2. CHECKING FOR LINKING KEYS:")
    games_cols = set(games_df.columns)
    users_cols = set(users_df.columns)
    recs_cols = set(recommendations_df.columns)
    meta_cols = set(metadata_df.columns)

    print("Common columns between datasets:")
    print(f"Games & Users: {games_cols.intersection(users_cols)}")
    print(f"Games & Recommendations: {games_cols.intersection(recs_cols)}")
    print(f"Games & Metadata: {games_cols.intersection(meta_cols)}")
    print(f"Users & Recommendations: {users_cols.intersection(recs_cols)}")

    print("\n3. DATA QUALITY OVERVIEW:")
    print("Missing values in each dataset:")
    print(f"Games: {games_df.isnull().sum().sum()} missing values")
    print(f"Users: {users_df.isnull().sum().sum()} missing values")
    print(f"Recommendations: {recommendations_df.isnull().sum().sum()} missing values")
    print(f"Metadata: {metadata_df.isnull().sum().sum()} missing values")

    print("\n4. SAMPLE DATA INSPECTION:")
    if 'user_id' in users_df.columns and 'game_id' in recommendations_df.columns:
        print("Found user and game identifiers - good for collaborative filtering")
    
    rating_cols = []
    for col in recommendations_df.columns:
        if 'rating' in col.lower():
            rating_cols.append(col)
        elif 'score' in col.lower():
            rating_cols.append(col)  

    if rating_cols:
        print("Found rating/score data - good for building preference models")
    
    genre_cols = []
    for col in games_df.columns:
        if 'genre' in col.lower():
            genre_cols.append(col)
        elif 'category' in col.lower():
            genre_cols.append(col)
    
    for col in metadata_df.columns:
        if 'genre' in col.lower():
            genre_cols.append(col)
        elif 'category' in col.lower():
            genre_cols.append(col)
    
    if genre_cols:
        print("Found genre/category data - good for content-based filtering")

    if 'hours' in recommendations_df.columns:
        rating_col = 'hours'
        print("Using 'hours' as preference indicator")
    elif 'helpful' in recommendations_df.columns:
        rating_col = 'helpful'
        print("Using 'helpful' as preference indicator")
    else:
        recommendations_df = recommendations_df.copy()
        recommendations_df['rating'] = 1
        rating_col = 'rating'
        print("Created binary ratings (1 = user interacted with game)")

    user_item_matrix = recommendations_df.pivot_table(
        index='user_id',
        columns='app_id',
        values=rating_col,
        fill_value=0,
        aggfunc='mean'
    )
    
    print(f"Matrix created: {user_item_matrix.shape[0]} users * {user_item_matrix.shape[1]} games")

    # Helper to list available user IDs that can be used with recommend_games_for_user
    def get_available_user_ids(max_count=50):
        ids = list(user_item_matrix.index)
        return ids[:max_count]

    # Show a short preview so users know which IDs are valid
    preview_ids = get_available_user_ids(20)
    print(f"Sample available user IDs (from interactions data): {preview_ids} ...")

    def get_collaborative_recommendations(user_id, n_recommendations=5):
        print(f"\n Collaborative Filtering for User {user_id}: ")

        if user_id not in user_item_matrix.index:
            print(f" User {user_id} not found")
            return []
        
        # Get user ratings
        user_ratings = user_item_matrix.loc[user_id]

        # Compute similaritiy with all other users
        user_matrix = user_item_matrix.values
        user_row = user_ratings.values.reshape(1, -1)
        similarities = cosine_similarity(user_row, user_matrix)[0]

        # Find most similar users (excluding self)
        similar_user_indices = np.argsort(similarities)[-11:-1][::-1]

        # Get games that similar users liked but current user hasn't played
        played_games = set(user_ratings[user_ratings > 0].index)
        recommendations = {}

        for idx in similar_user_indices:
            similar_user_id = user_item_matrix.index[idx]
            similar_ratings = user_item_matrix.loc[similar_user_id]
            similarity_score = similarities[idx]

            for game_id, rating in similar_ratings.items():
                if rating > 0 and game_id not in played_games:
                    if game_id not in recommendations:
                        recommendations[game_id] = 0
                    recommendations[game_id] += rating * similarity_score
        
        # Sort and return top recommendations
        if recommendations:
            sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
            return sorted_recs[:n_recommendations]
        return []
    
    # Content-based filtering function
    def get_content_based_recommendations(user_id, n_recommendations=5):
        # Generate content based recommendations
        print(f"\n Content-based filtering for user {user_id}:")

        if len(metadata_df) == 0:
            print("No metadata available")
            return []
        
        if user_id not in user_item_matrix.index:
            print(f" User {user_id} not found")
            return []
        
        # Create content features from metadata
        game_features = []
        game_ids = []

        for _, game in metadata_df.iterrows():
            content = ""
            tags_value = game.get('tags') if 'tags' in metadata_df.columns else None
            if tags_value is not None:
                if isinstance(tags_value, (list, tuple, set)):
                    content += " ".join(map(str, tags_value)) + " "
                else:
                    content += str(tags_value) + " "
            desc_value = game.get('description') if 'description' in metadata_df.columns else None
            if desc_value is not None and not (isinstance(desc_value, float) and np.isnan(desc_value)):
                content += str(desc_value) + " "
            
            if content.strip():
                game_features.append(content.strip())
                if 'app_id' in game.index:
                    game_ids.append(game['app_id'])
        
        if not game_features:
            print("No usable content features found")
            return []

        # Create TF-IDF vectors
        tfidf = TfidfVectorizer(max_features=500, stop_words='english')
        tfidf_matrix = tfidf.fit_transform(game_features)

        # Get user's preferences
        user_ratings = user_item_matrix.loc[user_id]
        liked_games = user_ratings[user_ratings > 0].index.tolist()

        # Build user content profile
        user_profile = np.zeros(tfidf_matrix.shape[1])
        profile_count = 0

        for game_id in liked_games:
            if game_id in game_ids:
                game_idx = game_ids.index(game_id)
                user_profile += tfidf_matrix[game_idx].toarray()[0]
                profile_count += 1
        
        if profile_count == 0:
            print("No content data for user's games")
            # Fallback: popularity-based recommendations with normalized scores
            candidate_df = games_df.copy()
            if 'app_id' in candidate_df.columns:
                candidate_df = candidate_df[~candidate_df['app_id'].isin(liked_games)]

            # Prepare normalized components
            candidate_df = candidate_df.copy()
            if 'user_reviews' in candidate_df.columns and candidate_df['user_reviews'].max() > candidate_df['user_reviews'].min():
                ur_min = float(candidate_df['user_reviews'].min())
                ur_max = float(candidate_df['user_reviews'].max())
                candidate_df['pop_reviews'] = (candidate_df['user_reviews'] - ur_min) / (ur_max - ur_min)
            else:
                candidate_df['pop_reviews'] = 0.0

            if 'positive_ratio' in candidate_df.columns:
                candidate_df['pop_ratio'] = candidate_df['positive_ratio'].astype(float) / 100.0
            else:
                candidate_df['pop_ratio'] = 0.0

            # Weighted popularity score (tuneable weights)
            candidate_df['popularity_score'] = 0.7 * candidate_df['pop_reviews'] + 0.3 * candidate_df['pop_ratio']

            # Sort by score and return
            candidate_df = candidate_df.sort_values(by=['popularity_score'], ascending=False)
            top = candidate_df.head(n_recommendations)
            fallback = []
            for _, row in top.iterrows():
                app_id_value = int(row['app_id']) if 'app_id' in row else row.name
                fallback.append((app_id_value, float(row['popularity_score'])))
            return fallback
        
        user_profile = user_profile / profile_count

        # Find similar games
        similarities = cosine_similarity([user_profile], tfidf_matrix)[0]
        played_games = set(liked_games)

        recommendations = []
        for i, sim in enumerate(similarities):
            game_id = game_ids[i]
            if game_id not in played_games:
                recommendations.append((game_id, sim))

        # sort and return top recommendations
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]
    
    # Get game information function
    def get_game_name(app_id):
        # Get game name from available data
        if 'name' in games_df.columns and 'app_id' in games_df.columns:
            game_matches = games_df['app_id'] == app_id
            if game_matches.any():
                game_info = games_df[game_matches]
                return game_info['name'].iloc[0]
            
        # Try other name columns
        for col in games_df.columns:
            if 'name' in col.lower() or 'title' in col.lower():
                if 'app_id' in games_df.columns:
                    game_matches = games_df['app_id'] == app_id
                    if game_matches.any():
                        game_info = games_df[game_matches]
                        return str(game_info[col].iloc[0])
                else:
                    # If no app_id column, return first game name
                    if len(games_df) > 0:
                        return str(games_df[col].iloc[0])
                
        return f"Game {app_id}"
    
    # Main recommendation function
    def recommend_games_for_user(user_id, n_recommendations=5, silent=False):
        # Generate recommendations for a user
        if not silent:
            print(f"\n Generating Recommendations for User {user_id}")
        
        # Get collaborative recommendations
        collab_recs = get_collaborative_recommendations(user_id, n_recommendations)

        # Get content-based recommendations
        content_recs = get_content_based_recommendations(user_id, n_recommendations)

        # Display results
        if not silent:
            print(f"\n Recommendations:")
            
            if collab_recs:
                print("\n Collaborative filtering (similar users also liked):")
                for i, (app_id, score) in enumerate(collab_recs, 1):
                    game_name = get_game_name(app_id)
                    print(f"   {i}.  {game_name} (Score: {score:.3f})")
            else:
                print(f"\n Content-based: No recommendations found")
            
            if content_recs:
                print("\nContent-based filtering (based on your game prefrerences):")
                for i, (app_id, score) in enumerate(content_recs, 1):
                    game_name = get_game_name(app_id)
                    print(f". {i}. {game_name} (Score: {score:.3f})")
            else:
                print("\nContent-based: No recommendations found")
            
        return collab_recs, content_recs

    def print_recommendations_clean(user_id, collab_recs, content_recs):
        all_rows = []
        rank = 1
        for method, recs in (("Collaborative", collab_recs or []), ("Content", content_recs or [])):
            for app_id, score in recs:
                all_rows.append({
                    "Rank": rank,
                    "Method": method,
                    "App ID": app_id,
                    "Game": get_game_name(app_id),
                    "Score": f"{score:.3f}",
                })
                rank += 1
        if not all_rows:
            print(f"No recommendations for user {user_id}.")
            return
        df = pd.DataFrame(all_rows, columns=["Rank", "Method", "Game", "App ID", "Score"])
        print("\n=== Recommendations ===")
        print(df.to_string(index=False))
        
    # --- CLI interface ---
    parser = argparse.ArgumentParser(description="Steam Game Recommender")
    parser.add_argument("--user-id", type=int, help="User ID to get recommendations for")
    parser.add_argument("--top", type=int, default=5, help="Number of recommendations to show")
    parser.add_argument("--verbose", action="store_true", help="Show detailed dataset prints")
    parser.add_argument("--list-users", action="store_true", help="List sample available user IDs and exit")
    parser.add_argument("--save", type=str, default=None, help="Path to save recommendations CSV")
    args, unknown = parser.parse_known_args()

    if args.verbose and not VERBOSE_OUTPUT:
        print("Tip: Set VERBOSE_OUTPUT = True at top of file for persistent verbosity.")

    if args.list_users:
        ids = list(user_item_matrix.index[:50])
        print("First 50 available user IDs:", ids)
        raise SystemExit(0)

    # Choose a user: either provided or smart-picked sample
    target_user = args.user_id
    if target_user is None:
        print("\n Demonstration:")
        # Pick a sample user who has at least one game with available metadata
        available_game_ids = set()
        temp_game_ids = []
        for _, game in metadata_df.iterrows():
            has_content = False
            if 'tags' in metadata_df.columns and game.get('tags') is not None:
                has_content = True
            if 'description' in metadata_df.columns and game.get('description') is not None:
                has_content = True
            if has_content and 'app_id' in game.index:
                temp_game_ids.append(game['app_id'])
        available_game_ids = set(temp_game_ids)

        for uid in user_item_matrix.index:
            user_games_series = user_item_matrix.loc[uid]
            liked = set(user_games_series[user_games_series > 0].index)
            if liked & available_game_ids:
                target_user = uid
                break
        if target_user is None:
            target_user = user_item_matrix.index[0]

    print(f"\nGetting recommendations for user: {target_user}")
    collab_recs, content_recs = recommend_games_for_user(target_user, n_recommendations=args.top, silent=True)
    print_recommendations_clean(target_user, collab_recs, content_recs)

    if args.save is not None:
        rows = []
        for app_id, score in (collab_recs or []):
            rows.append({"method": "collaborative", "app_id": app_id, "score": score, "name": get_game_name(app_id)})
        for app_id, score in (content_recs or []):
            rows.append({"method": "content", "app_id": app_id, "score": score, "name": get_game_name(app_id)})
        if rows:
            pd.DataFrame(rows).to_csv(args.save, index=False)
            print(f"Saved recommendations to {args.save}")
        else:
            print("No recommendations to save.")

    print(f"\n Dataset Stats:")
    print(f" {len(user_item_matrix.index)} users")
    print(f" {len(user_item_matrix.columns)} games")
    print(f" {len(recommendations_df)} interactions")
    print(f"\nTo get recommendations for any user:")
    print(f" python product_recommendation_project.py --user-id <ID> --top 5")
    print(f"\nSample available users: {list(user_item_matrix.index[:5])}...")

except FileNotFoundError as e:
    print(f"Error: The file {e.filename} was not found.:")
    print("Please make sure the file exists in the same directory.")
except pd.errors.EmptyDataError as e:
    print(f"Error: One of the CSV files is empty or corrupted: {e}")
except json.JSONDecodeError as e:
    print(f"Error: JSON file is malformed: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
    print(f"Error type: {type(e).__name__}")