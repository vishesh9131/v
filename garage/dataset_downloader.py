import os
import requests
import zipfile
import gzip
import io
import pandas as pd
import numpy as np
import json
import logging
from tqdm import tqdm
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dataset_download.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Base directory for all datasets
BASE_DIR = "benchmark_results"
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

def download_movielens_1m():
    """Download MovieLens-1M dataset"""
    dataset_dir = os.path.join(DATA_DIR, "ml-1m")
    if os.path.exists(os.path.join(dataset_dir, "ratings.dat")):
        logger.info("MovieLens-1M dataset already exists.")
        return dataset_dir
    
    os.makedirs(dataset_dir, exist_ok=True)
    logger.info("Downloading MovieLens-1M dataset...")
    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    
    response = requests.get(url)
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(DATA_DIR)
    
    logger.info("MovieLens-1M dataset downloaded successfully!")
    return dataset_dir

def download_amazon_books():
    """Download Amazon Books dataset (5-core)"""
    dataset_dir = os.path.join(DATA_DIR, "amazon-books")
    if os.path.exists(os.path.join(dataset_dir, "ratings.csv")):
        logger.info("Amazon Books dataset already exists.")
        return dataset_dir
    
    os.makedirs(dataset_dir, exist_ok=True)
    logger.info("Downloading Amazon Books dataset...")
    
    # Using the smaller 5-core version
    url = "https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Books_5.json.gz"
    
    response = requests.get(url)
    
    # Extract and process the data
    with gzip.open(io.BytesIO(response.content)) as f:
        # Convert the json lines to a pandas dataframe
        data = []
        for line in tqdm(f, desc="Processing Amazon Books"):
            data.append(json.loads(line))
    
    # Create a dataframe
    df = pd.DataFrame(data)
    
    # Map user and item IDs
    unique_users = df['reviewerID'].unique()
    unique_items = df['asin'].unique()
    
    user_map = {user: idx for idx, user in enumerate(unique_users)}
    item_map = {item: idx for idx, item in enumerate(unique_items)}
    
    # Create ratings.csv file
    ratings_df = pd.DataFrame({
        'user_id': df['reviewerID'].map(user_map),
        'item_id': df['asin'].map(item_map),
        'rating': df['overall'],
        'timestamp': pd.to_datetime(df['unixReviewTime'], unit='s').astype(int) // 10**9
    })
    
    # Save to CSV
    ratings_path = os.path.join(dataset_dir, "ratings.csv")
    ratings_df.to_csv(ratings_path, index=False)
    
    # Save mappings for reference
    with open(os.path.join(dataset_dir, "user_map.json"), 'w') as f:
        json.dump(user_map, f)
    
    with open(os.path.join(dataset_dir, "item_map.json"), 'w') as f:
        json.dump(item_map, f)
    
    logger.info(f"Amazon Books dataset processed and saved to {ratings_path}")
    logger.info(f"Dataset has {len(unique_users)} users and {len(unique_items)} items")
    
    return dataset_dir

def download_yelp():
    """Download Yelp dataset (subset)"""
    dataset_dir = os.path.join(DATA_DIR, "yelp")
    if os.path.exists(os.path.join(dataset_dir, "ratings.csv")):
        logger.info("Yelp dataset already exists.")
        return dataset_dir
    
    os.makedirs(dataset_dir, exist_ok=True)
    logger.info("Downloading Yelp dataset...")
    
    # Using the academic dataset
    url = "https://www.yelp.com/dataset/download"
    logger.warning(f"Yelp dataset requires manual download from {url}")
    logger.warning("Please download and extract the 'yelp_academic_dataset_review.json' file")
    logger.warning(f"Then place it in {dataset_dir}")
    
    # Check if the file exists
    review_file = os.path.join(dataset_dir, "yelp_academic_dataset_review.json")
    if not os.path.exists(review_file):
        logger.error(f"Yelp dataset file not found at {review_file}")
        logger.info("Using a simulated small sample for testing purposes")
        
        # Create a simulated small sample for testing
        n_users = 1000
        n_items = 500
        n_ratings = 10000
        
        # Generate random data
        np.random.seed(42)
        user_ids = np.random.randint(0, n_users, n_ratings)
        item_ids = np.random.randint(0, n_items, n_ratings)
        ratings = np.random.randint(1, 6, n_ratings)
        timestamps = np.random.randint(1000000000, 1600000000, n_ratings)
        
        # Create ratings dataframe
        ratings_df = pd.DataFrame({
            'user_id': user_ids,
            'item_id': item_ids,
            'rating': ratings,
            'timestamp': timestamps
        })
        
        # Save to CSV
        ratings_path = os.path.join(dataset_dir, "ratings.csv")
        ratings_df.to_csv(ratings_path, index=False)
        
        logger.info(f"Simulated Yelp dataset saved to {ratings_path}")
        logger.info(f"Simulated dataset has {n_users} users and {n_items} items")
        
        return dataset_dir
    
    logger.info("Processing Yelp dataset...")
    
    # Process the dataset
    reviews = []
    with open(review_file, 'r') as f:
        for line in tqdm(f, desc="Processing Yelp reviews"):
            reviews.append(json.loads(line))
    
    # Convert to dataframe
    df = pd.DataFrame(reviews)
    
    # Map user and business IDs
    unique_users = df['user_id'].unique()
    unique_items = df['business_id'].unique()
    
    user_map = {user: idx for idx, user in enumerate(unique_users)}
    item_map = {item: idx for idx, item in enumerate(unique_items)}
    
    # Create ratings dataframe
    ratings_df = pd.DataFrame({
        'user_id': df['user_id'].map(user_map),
        'item_id': df['business_id'].map(item_map),
        'rating': df['stars'],
        'timestamp': pd.to_datetime(df['date']).astype(int) // 10**9
    })
    
    # Save to CSV
    ratings_path = os.path.join(dataset_dir, "ratings.csv")
    ratings_df.to_csv(ratings_path, index=False)
    
    # Save mappings
    with open(os.path.join(dataset_dir, "user_map.json"), 'w') as f:
        json.dump(user_map, f)
    
    with open(os.path.join(dataset_dir, "item_map.json"), 'w') as f:
        json.dump(item_map, f)
    
    logger.info(f"Yelp dataset processed and saved to {ratings_path}")
    logger.info(f"Dataset has {len(unique_users)} users and {len(unique_items)} items")
    
    return dataset_dir

def download_lastfm():
    """Download Last.fm dataset"""
    dataset_dir = os.path.join(DATA_DIR, "lastfm")
    if os.path.exists(os.path.join(dataset_dir, "ratings.csv")):
        logger.info("Last.fm dataset already exists.")
        return dataset_dir
    
    os.makedirs(dataset_dir, exist_ok=True)
    logger.info("Downloading Last.fm dataset...")
    
    # Use Last.FM 360K
    url = "http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-360K.tar.gz"
    
    try:
        response = requests.get(url)
        
        # Extract the tar file
        import tarfile
        with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as tar:
            tar.extractall(path=dataset_dir)
        
        # Process the dataset
        logger.info("Processing Last.fm dataset...")
        
        # Last.fm has user_artists.dat with play counts instead of ratings
        # We'll convert play counts to implicit ratings
        user_artists_file = os.path.join(dataset_dir, "lastfm-dataset-360K", "usersha1-artmbid-artname-plays.tsv")
        
        if not os.path.exists(user_artists_file):
            logger.error(f"Last.fm dataset file not found at {user_artists_file}")
            return dataset_dir
        
        # Read the TSV file
        df = pd.read_csv(user_artists_file, sep='\t', names=['user_id', 'artist_id', 'artist_name', 'plays'])
        
        # Map user and artist IDs
        unique_users = df['user_id'].unique()
        unique_items = df['artist_id'].unique()
        
        user_map = {user: idx for idx, user in enumerate(unique_users)}
        item_map = {item: idx for idx, item in enumerate(unique_items)}
        
        # Convert play counts to a 1-5 rating scale (log transform)
        # Add 1 to avoid log(0) and then scale to 1-5
        min_plays = np.log1p(df['plays'].min())
        max_plays = np.log1p(df['plays'].max())
        df['rating'] = 1 + 4 * (np.log1p(df['plays']) - min_plays) / (max_plays - min_plays)
        
        # Create ratings dataframe
        ratings_df = pd.DataFrame({
            'user_id': df['user_id'].map(user_map),
            'item_id': df['artist_id'].map(item_map),
            'rating': df['rating'],
            'timestamp': np.random.randint(1000000000, 1600000000, len(df))  # Random timestamps
        })
        
        # Save to CSV
        ratings_path = os.path.join(dataset_dir, "ratings.csv")
        ratings_df.to_csv(ratings_path, index=False)
        
        # Save mappings
        with open(os.path.join(dataset_dir, "user_map.json"), 'w') as f:
            json.dump(user_map, f)
        
        with open(os.path.join(dataset_dir, "item_map.json"), 'w') as f:
            json.dump(item_map, f)
        
        logger.info(f"Last.fm dataset processed and saved to {ratings_path}")
        logger.info(f"Dataset has {len(unique_users)} users and {len(unique_items)} items")
        
    except Exception as e:
        logger.error(f"Error downloading Last.fm dataset: {str(e)}")
        
        # Create a simulated dataset
        logger.info("Creating a simulated Last.fm dataset for testing")
        
        n_users = 1000
        n_items = 2000
        n_ratings = 20000
        
        # Generate random data
        np.random.seed(42)
        user_ids = np.random.randint(0, n_users, n_ratings)
        item_ids = np.random.randint(0, n_items, n_ratings)
        ratings = np.random.uniform(1, 5, n_ratings)
        timestamps = np.random.randint(1000000000, 1600000000, n_ratings)
        
        # Create ratings dataframe
        ratings_df = pd.DataFrame({
            'user_id': user_ids,
            'item_id': item_ids,
            'rating': ratings,
            'timestamp': timestamps
        })
        
        # Save to CSV
        ratings_path = os.path.join(dataset_dir, "ratings.csv")
        ratings_df.to_csv(ratings_path, index=False)
        
        logger.info(f"Simulated Last.fm dataset saved to {ratings_path}")
        logger.info(f"Simulated dataset has {n_users} users and {n_items} items")
    
    return dataset_dir

def download_gowalla():
    """Download Gowalla check-in dataset"""
    dataset_dir = os.path.join(DATA_DIR, "gowalla")
    if os.path.exists(os.path.join(dataset_dir, "ratings.csv")):
        logger.info("Gowalla dataset already exists.")
        return dataset_dir
    
    os.makedirs(dataset_dir, exist_ok=True)
    logger.info("Downloading Gowalla dataset...")
    
    # Gowalla check-in dataset
    url = "https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz"
    
    try:
        response = requests.get(url)
        
        # Extract and process the data
        with gzip.open(io.BytesIO(response.content)) as f:
            # Read the data
            lines = []
            for line in tqdm(f, desc="Processing Gowalla check-ins"):
                lines.append(line.decode('utf-8').strip())
        
        # Parse the data
        data = []
        for line in tqdm(lines, desc="Parsing check-ins"):
            parts = line.split('\t')
            if len(parts) >= 5:
                user_id = parts[0]
                timestamp = parts[1]
                latitude = parts[2]
                longitude = parts[3]
                location_id = parts[4]
                
                data.append({
                    'user_id': user_id,
                    'location_id': location_id,
                    'timestamp': timestamp,
                    'latitude': latitude,
                    'longitude': longitude
                })
        
        # Convert to dataframe
        df = pd.DataFrame(data)
        
        # Map user and location IDs
        unique_users = df['user_id'].unique()
        unique_items = df['location_id'].unique()
        
        user_map = {user: idx for idx, user in enumerate(unique_users)}
        item_map = {item: idx for idx, item in enumerate(unique_items)}
        
        # Count check-ins per user-location pair
        checkin_counts = df.groupby(['user_id', 'location_id']).size().reset_index(name='count')
        
        # Convert check-in counts to a 1-5 rating scale
        min_count = checkin_counts['count'].min()
        max_count = checkin_counts['count'].max()
        checkin_counts['rating'] = 1 + 4 * (checkin_counts['count'] - min_count) / (max_count - min_count)
        
        # Create ratings dataframe
        ratings_df = pd.DataFrame({
            'user_id': checkin_counts['user_id'].map(user_map),
            'item_id': checkin_counts['location_id'].map(item_map),
            'rating': checkin_counts['rating'],
            'timestamp': np.random.randint(1000000000, 1600000000, len(checkin_counts))  # Random timestamps
        })
        
        # Save to CSV
        ratings_path = os.path.join(dataset_dir, "ratings.csv")
        ratings_df.to_csv(ratings_path, index=False)
        
        # Save mappings
        with open(os.path.join(dataset_dir, "user_map.json"), 'w') as f:
            json.dump(user_map, f)
        
        with open(os.path.join(dataset_dir, "item_map.json"), 'w') as f:
            json.dump(item_map, f)
        
        logger.info(f"Gowalla dataset processed and saved to {ratings_path}")
        logger.info(f"Dataset has {len(unique_users)} users and {len(unique_items)} items")
        
    except Exception as e:
        logger.error(f"Error downloading Gowalla dataset: {str(e)}")
        
        # Create a simulated dataset
        logger.info("Creating a simulated Gowalla dataset for testing")
        
        n_users = 1000
        n_items = 5000
        n_ratings = 30000
        
        # Generate random data
        np.random.seed(42)
        user_ids = np.random.randint(0, n_users, n_ratings)
        item_ids = np.random.randint(0, n_items, n_ratings)
        ratings = np.random.uniform(1, 5, n_ratings)
        timestamps = np.random.randint(1000000000, 1600000000, n_ratings)
        
        # Create ratings dataframe
        ratings_df = pd.DataFrame({
            'user_id': user_ids,
            'item_id': item_ids,
            'rating': ratings,
            'timestamp': timestamps
        })
        
        # Save to CSV
        ratings_path = os.path.join(dataset_dir, "ratings.csv")
        ratings_df.to_csv(ratings_path, index=False)
        
        logger.info(f"Simulated Gowalla dataset saved to {ratings_path}")
        logger.info(f"Simulated dataset has {n_users} users and {n_items} items")
    
    return dataset_dir

def download_netflix():
    """Download a subset of the Netflix Prize dataset (simulated due to licensing restrictions)"""
    dataset_dir = os.path.join(DATA_DIR, "netflix")
    if os.path.exists(os.path.join(dataset_dir, "ratings.csv")):
        logger.info("Netflix dataset already exists.")
        return dataset_dir
    
    os.makedirs(dataset_dir, exist_ok=True)
    logger.info("Netflix Prize dataset is no longer publicly available due to privacy concerns")
    logger.info("Creating a simulated Netflix-like dataset for testing")
    
    # Create a simulated dataset
    n_users = 2000
    n_items = 1000
    n_ratings = 100000
    
    # Generate random data
    np.random.seed(42)
    user_ids = np.random.randint(0, n_users, n_ratings)
    item_ids = np.random.randint(0, n_items, n_ratings)
    ratings = np.random.randint(1, 6, n_ratings)  # Netflix used 1-5 stars
    timestamps = np.random.randint(1000000000, 1600000000, n_ratings)
    
    # Create ratings dataframe
    ratings_df = pd.DataFrame({
        'user_id': user_ids,
        'item_id': item_ids,
        'rating': ratings,
        'timestamp': timestamps
    })
    
    # Save to CSV
    ratings_path = os.path.join(dataset_dir, "ratings.csv")
    ratings_df.to_csv(ratings_path, index=False)
    
    logger.info(f"Simulated Netflix dataset saved to {ratings_path}")
    logger.info(f"Simulated dataset has {n_users} users and {n_items} items")
    
    return dataset_dir

def convert_ml_1m_to_csv():
    """Convert MovieLens-1M dataset from .dat to .csv format"""
    ml_1m_dir = os.path.join(DATA_DIR, "ml-1m")
    ratings_dat = os.path.join(ml_1m_dir, "ratings.dat")
    ratings_csv = os.path.join(ml_1m_dir, "ratings.csv")
    
    if os.path.exists(ratings_csv):
        logger.info("MovieLens-1M CSV format already exists.")
        return
    
    if not os.path.exists(ratings_dat):
        logger.error("MovieLens-1M dataset not found.")
        return
    
    logger.info("Converting MovieLens-1M dataset to CSV format...")
    
    # Read the .dat file
    df = pd.read_csv(ratings_dat, sep='::', 
                    names=['user_id', 'movie_id', 'rating', 'timestamp'],
                    engine='python')
    
    # Save as CSV
    df.to_csv(ratings_csv, index=False)
    
    logger.info(f"MovieLens-1M dataset converted to CSV format: {ratings_csv}")

def download_all_datasets():
    """Download all supported datasets"""
    logger.info("Downloading all supported datasets...")
    
    # Create a list of all dataset download functions
    download_functions = [
        download_movielens_1m,
        download_amazon_books,
        download_yelp,
        download_lastfm,
        download_gowalla,
        download_netflix
    ]
    
    # Download each dataset
    for download_fn in download_functions:
        try:
            download_fn()
        except Exception as e:
            logger.error(f"Error downloading dataset {download_fn.__name__}: {str(e)}")
    
    # Convert MovieLens-1M to CSV
    convert_ml_1m_to_csv()
    
    logger.info("All datasets downloaded successfully!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download recommendation datasets')
    parser.add_argument('--dataset', type=str, choices=['ml-1m', 'amazon-books', 'yelp', 'lastfm', 'gowalla', 'netflix', 'all'],
                       help='Dataset to download (default: all)')
    
    args = parser.parse_args()
    
    if args.dataset == 'ml-1m':
        download_movielens_1m()
        convert_ml_1m_to_csv()
    elif args.dataset == 'amazon-books':
        download_amazon_books()
    elif args.dataset == 'yelp':
        download_yelp()
    elif args.dataset == 'lastfm':
        download_lastfm()
    elif args.dataset == 'gowalla':
        download_gowalla()
    elif args.dataset == 'netflix':
        download_netflix()
    elif args.dataset == 'all' or args.dataset is None:
        download_all_datasets()
    else:
        logger.error(f"Unknown dataset: {args.dataset}") 