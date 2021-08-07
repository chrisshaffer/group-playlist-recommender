from load_msd_subset import *
import numpy as np
import pandas as pd

def tt_preprocess():
    # Load data into pd df
    tt = load_train_triplets()

    # Convert playcounts > 100 to 100
    tt.plays = tt.plays.transform(lambda x: 100 if x > 100 else x)

    # Create 'rating' column of log10(plays)/max(log10(plays))
    max_log_plays = np.max(np.log10(tt['plays']))
    tt['rating'] = np.log10(tt.plays)/max_log_plays
    
    # Include only top 1000 tracks (20% of total data, ~10 millions entries)
    top_tracks = tt['track_id'].value_counts()[:100].index.tolist()
    tt = tt[tt['track_id'].isin(top_tracks)]

    # Drop plays
    tt = tt.drop(columns='plays')

    # Create dummy column, timestamp, to fit Surprise format
    tt['timestamp'] = 0
    
    return tt