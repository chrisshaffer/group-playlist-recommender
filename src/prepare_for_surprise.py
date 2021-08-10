from load_msd_subset import *
import numpy as np
import pandas as pd

def tt_preprocess():
    # Load data into pd df
    tt = load_train_triplets()

    # Saturate top 1% of plays, where 24 is 99th percentile of plays
    tt.plays = tt.plays.transform(lambda x: 24 if x > 24 else x)

    # Create 'rating' column based on log(plays/max_plays) transformed to 1-5 scale
    tt['rating'] = np.log10(tt.plays)/np.log10(tt['plays'].max())*5+1
    
    # Include only top 250 tracks (11% of total data, ~4.8 millions entries)
    top_tracks = tt['track_id'].value_counts()[:250].index.tolist()
    tt = tt[tt['track_id'].isin(top_tracks)]

    # Drop plays
    tt = tt.drop(columns='plays')

    # Create dummy column, timestamp, to fit Surprise format
    tt['timestamp'] = 0
    
    return tt