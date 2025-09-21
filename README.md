Overview
A machine learning-powered recommendation system that suggests games to Steam users based on their review history and gameplay patterns. The system uses both collaborative filtering (finding similar users) and content-based filtering (analyzing game features) to provide personalized game recommendations.
Features

Collaborative Filtering: Recommends games based on similar users' preferences
Content-Based Filtering: Suggests games with similar tags, descriptions, and features
Hybrid Approach: Combines both methods for comprehensive recommendations
Command Line Interface: Easy-to-use CLI with various options
Data Export: Save recommendations to CSV files
User Analysis: Inspect available users and their gaming patterns

Data Requirements
The system requires four data files from Kaggle's Steam dataset:

games.csv - Game information and metadata
users.csv - User profile data
recommendations.csv - User-game interactions and ratings
games_metadata.json - Detailed game metadata including tags and descriptions

Note: Data files are not included in this repository. Download them from Kaggle's Steam dataset and place them in the project directory.
Installation

Clone this repository:

bashgit clone https://github.com/jesskw21/steam-recommendation-engine.git
cd steam-recommendation-engine

Install required dependencies:

bashpip install pandas numpy scikit-learn

Download the required data files and place them in the project directory.

Usage
Basic Usage
Run the recommendation engine with default settings:
bashpython product_recommendation_project.py
Command Line Options
bash# Get recommendations for a specific user
python product_recommendation_project.py --user-id 12345 --top 10

# List available user IDs
python product_recommendation_project.py --list-users

# Save recommendations to CSV
python product_recommendation_project.py --user-id 12345 --save recommendations_output.csv

# Enable verbose output for detailed dataset information
python product_recommendation_project.py --verbose
Available Arguments

--user-id: Specify a user ID for personalized recommendations
--top: Number of recommendations to generate (default: 5)
--list-users: Display sample available user IDs
--save: Save recommendations to specified CSV file
--verbose: Show detailed dataset exploration output

How It Works
1. Data Processing

Loads and validates all required data files
Creates user-item interaction matrix from ratings/gameplay hours
Processes game metadata for content-based features

2. Collaborative Filtering

Calculates user similarity using cosine similarity
Identifies users with similar gaming preferences
Recommends games that similar users enjoyed

3. Content-Based Filtering

Analyzes game tags, descriptions, and metadata
Creates TF-IDF vectors for game content features
Builds user preference profiles based on played games
Suggests games with similar content characteristics

4. Fallback Mechanisms

Popularity-based recommendations when user data is insufficient
Handles missing data gracefully with normalized scoring

Output Example
=== Recommendations ===
Rank Method      Game                           App ID    Score
1    Collaborative Counter-Strike: Global Offensive 730      0.856
2    Collaborative Dota 2                          570      0.743
3    Content      Portal 2                       620      0.692
4    Content      The Witcher 3: Wild Hunt       292030   0.654
5    Collaborative Team Fortress 2               440      0.621
Technical Details
Dependencies

pandas: Data manipulation and analysis
numpy: Numerical computing
scikit-learn: Machine learning algorithms (TF-IDF, cosine similarity)
json: JSON data processing

Key Components

User-Item Matrix: Sparse matrix of user-game interactions
TF-IDF Vectorization: Content feature extraction from game metadata
Cosine Similarity: Measuring user and content similarity
Hybrid Scoring: Combining collaborative and content-based scores

Data Quality Handling

Automatically detects and reports missing files
Handles JSON parsing errors gracefully
Manages missing values in datasets
Provides fallback recommendations when primary methods fail

Limitations

Requires substantial user interaction data for collaborative filtering
Content-based filtering depends on quality of game metadata
Cold start problem for new users with no gaming history
Performance scales with dataset size

Future Enhancements

Matrix factorization techniques (SVD, NMF)
Deep learning approaches for feature extraction
Real-time recommendation updates
A/B testing framework for recommendation quality
Web interface for easier user interaction

Contributing
Contributions are welcome! Areas for improvement:

Additional recommendation algorithms
Performance optimizations
Enhanced data preprocessing
Better evaluation metrics

License
MIT License - See LICENSE file for details
Acknowledgments

Kaggle for providing the Steam dataset
Steam community for game reviews and metadata
Scikit-learn team for machine learning tools

Contact
For questions about the recommendation system or suggestions for improvements, please open an issue in this repository.
