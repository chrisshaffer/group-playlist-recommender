import numpy as np
import pandas as pd

def load_MSS():
    
    # load MillionSongSubset into pandas DataFrame
    df = pd.read_csv('data/MSS.csv')
    numeric_cols = ['artist_latitude', 'artist_longitude', 'year', 'artist_hotttnesss', 
                    'artist_familiarity', 'duration', 'key', 'key_confidence', 
                    'loudness', 'mode', 'mode_confidence', 'end_of_fade_in', 
                    'start_of_fade_out', 'tempo', 'time_signature', 
                    'time_signature_confidence','track_id']
    
    df = df[numeric_cols]

    # # Drop index column
    # df.drop(columns='Unnamed: 0',inplace=True)
    
    # # Remove extraneous 'b from strings
    # b_remove = ['artist_id','artist_mbid','artist_location','track_id','artist_location',
    #         'artist_name']
    # for col in b_remove:
    #     df[col] = df[col].transform(lambda x: x[2:-1])
        
    # # Include only numeric columns + track_id column
    # is_number = np.vectorize(lambda x: np.issubdtype(x, np.number))
    # criteria = is_number(df.dtypes)
    # df.loc[:, criteria | (df.columns == 'track_id')]
    
    # Set track_id as index
    df.set_index('track_id',inplace=True)
    
    return df

def load_train_triplets():
    
    # Load train_triplets into pd DataFrame
    # Train triplets are from Taste Profiles, including user_id, track_id, and # of plays
    tt = pd.read_csv('data/train_triplets.txt',delimiter='\t', header=None)
    tt.columns = ['user_id', 'track_id','plays']
    
    return tt

def load_jam_data(jam_msd = False, jams = False, likes = False, followers = False):

    # Load DataFrame of jam_id and song_id correspondence
    if jam_msd:
        jam_msd = pd.read_csv('data/jam_to_msd.tsv',delimiter='\t')
        jam_msd.columns = ['jam_id','track_id']
        jam_msd.set_index('track_id',inplace=True)
    else:
        jam_msd = []

    # Data related to each ThisIsMyJam entry
    if jams:
        jams = pd.read_csv('data/jams/jams.tsv', sep = '\t',error_bad_lines=False,engine='python')
    else:
        jams = []
    
    # Load DataFrame of jam_id and likes correspondence 
    if likes:
        likes = pd.read_csv('data/jams/likes.tsv',delimiter='\t')
    else:
        likes = []
        
    # Load DataFrame of user_id and user_id for follows 
    if followers:
        followers = pd.read_csv('data/jams/followers.tsv',delimiter='\t')
    else:
        followers = []
    
    return jam_msd, jams, likes, followers

if __name__ == "__main__":
    
    df = load_MSS()
    tt = load_train_triplets()