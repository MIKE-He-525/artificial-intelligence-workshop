import argparse
import json
import numpy as np

def build_arg_parser():
    """Construct a command-line argument parser to get the target user name"""
    parser = argparse.ArgumentParser(description='Movie Recommendation System based on Collaborative Filtering')
    parser.add_argument("-user", dest="input_user", required=True, 
                        help='Input user name to get movie recommendations')
    return parser

def pearson_score(dataset, user1, user2):
    """
    Calculate the Pearson similarity score between two users (reuses logic from Section 6.4)
    :param dataset: Complete rating dataset (dictionary)
    :param user1: Name of the first user
    :param user2: Name of the second user
    :return: Pearson similarity score (-1 to 1); returns 0 if no common movies
    """
    # Check if users exist in the dataset
    if user1 not in dataset:
        raise TypeError(f"Cannot find {user1} in the dataset")
    if user2 not in dataset:
        raise TypeError(f"Cannot find {user2} in the dataset")
    
    # Extract movies rated by both users
    common_movies = {}
    for movie in dataset[user1]:
        if movie in dataset[user2]:
            common_movies[movie] = 1
    
    # Return 0 if there are no common movies
    num_common = len(common_movies)
    if num_common == 0:
        return 0
    
    # Calculate statistical metrics for ratings of common movies
    user1_ratings = [dataset[user1][movie] for movie in common_movies]
    user2_ratings = [dataset[user2][movie] for movie in common_movies]
    
    # Pearson formula: r = Sxy / sqrt(Sxx * Syy)
    user1_mean = np.mean(user1_ratings)
    user2_mean = np.mean(user2_ratings)
    
    Sxy = np.sum([(r1 - user1_mean) * (r2 - user2_mean) 
                  for r1, r2 in zip(user1_ratings, user2_ratings)])
    Sxx = np.sum([(r - user1_mean) ** 2 for r in user1_ratings])
    Syy = np.sum([(r - user2_mean) ** 2 for r in user2_ratings])
    
    # Avoid division by zero (when there is no rating variation)
    if Sxx * Syy == 0:
        return 0
    
    return Sxy / np.sqrt(Sxx * Syy)

def get_recommendations(dataset, input_user):
    """
    Generate movie recommendations for the target user
    :param dataset: Complete rating dataset (dictionary)
    :param input_user: Target user name
    :return: Sorted list of recommended movies; empty list if no recommendations
    """
    # Check if the target user exists
    if input_user not in dataset:
        raise TypeError(f"Cannot find {input_user} in the dataset")
    
    # Initialize dictionaries to track weighted ratings and total similarity scores
    overall_scores = {}  # key: movie name, value: weighted rating (similarity × user rating)
    similarity_scores = {}  # key: movie name, value: sum of similarities used for calculation
    
    # Iterate over all other users in the dataset to compute similarity and extract candidates
    for user in dataset:
        # Skip the target user themselves
        if user == input_user:
            continue
        
        # Calculate Pearson similarity between the target user and the current user
        sim_score = pearson_score(dataset, input_user, user)
        
        # Skip users with non-positive similarity (no positive contribution)
        if sim_score <= 0:
            continue
        
        # Extract movies rated by the current user but not by the target user (recommendation candidates)
        unrated_movies = [movie for movie in dataset[user] 
                          if movie not in dataset[input_user]]
        
        # Accumulate weighted ratings and similarities for candidate movies
        for movie in unrated_movies:
            overall_scores[movie] = overall_scores.get(movie, 0) + dataset[user][movie] * sim_score
            similarity_scores[movie] = similarity_scores.get(movie, 0) + sim_score
    
    # Return empty list if no candidate movies exist
    if not overall_scores:
        return []
    
    # Normalize ratings (weighted rating / total similarity) to eliminate similarity count bias
    movie_scores = [(movie, score / similarity_scores[movie]) 
                    for movie, score in overall_scores.items()]
    
    # Sort by rating in descending order to determine recommendation priority
    movie_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return a list of movie names sorted by recommendation priority
    return [movie for movie, score in movie_scores]

if __name__ == "__main__":
    # 1. Parse command-line arguments (get target user)
    args = build_arg_parser().parse_args()
    input_user = args.input_user
    
    # 2. Load rating data from ratings.json
    ratings_file = "ratings.json"
    try:
        with open(ratings_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"Error: {ratings_file} not found. Please check the file path.")
        exit(1)
    
    # 3. Generate recommendations and output results
    try:
        recommendations = get_recommendations(dataset, input_user)
    except TypeError as e:
        print(f"Error: {e}")
        exit(1)
    
    # 4. Print recommendation results
    print(f"\nMovie recommendations for {input_user}:")
    if recommendations:
        for rank, movie in enumerate(recommendations, start=1):
            print(f"{rank}. {movie}")
    else:
        print("No movie recommendations available (no similar users or all movies rated).")