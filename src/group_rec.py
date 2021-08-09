import pickle
import pandas as pd
from recommender import SongRecommender
import numpy as np
import warnings

warnings.filterwarnings("ignore")

class GroupRecommender():
    """Playlist recommender system for groups."""

    def __init__(self,user_ids,num_songs=5):
        """Constructs a GroupRecommender"""
        self.user_ids = user_ids
        self.num_songs = num_songs

    def score_for_users(self,train_path):
        self.train = pd.read_csv(train_path)
        
        model = pickle.load( open( "data/model.p", "rb" ) )

        track_list = self.train['track_id'].unique()
        
        df = pd.DataFrame(columns = self.user_ids, index = track_list).reset_index()
        
        request_data = pd.melt(df, id_vars = 'index', value_vars=user_ids)
        request_data.rename(columns={'index': 'track_id', 'variable': 'user_id', 'value': 'rating'},inplace=True)
        request_data = request_data[['user_id', 'track_id', 'rating']]
        request_data['timestamp'] = 0

        # Predict for request_data, returns a dataframe
        self.predictions = model.transform(request_data)
        
        return self
        
    def impute_knowns(self):
        updated = self.predictions.merge(self.train, how='left', on=['user_id', 'track_id'],
                            suffixes=('', '_new'))
        updated.drop(columns = ['Unnamed: 0','timestamp'],inplace=True)

        updated['rating'] = np.where(pd.notnull(updated['rating_new']), updated['rating_new'], updated['rating'])
        # # # Modify here
        # updated['known'] = np.where(pd.notnull(updated['rating_new']), updated['rating_new'], updated['rating'])
        updated.drop('rating_new', axis=1, inplace=True)
        
        self.predictions = updated
        
        return self
        
    def create_rankings(self):
        dfs = []
        for idx, user_id in enumerate(self.user_ids):
            sorted_by_rating = self.predictions[self.predictions['user_id'] == user_id].sort_values(
                by='rating',ascending=False).reset_index().drop(
                columns=['user_id','index']).reset_index().set_index('track_id')
            dfs.append(sorted_by_rating.rename(columns={'index':'rank'},inplace=True))
            if idx == 0:
                rankings = sorted_by_rating
            else:
                rankings = rankings.join(sorted_by_rating,rsuffix=str(idx+1))

        self.rankings = rankings
        
        return self
    
    def track_artist_names(self,df):
        """Extract song_title and artist_name for tracks in given DataFrame

        Args:
            df (pandas DataFrame): DataFrame of top recommended songs

        Returns:
            pandas DataFrame: Initial DataFrame with additional song_title, artist_name columns
        """    
        
        # read in map of track_id to artist_name and song_title
        names = pd.read_csv('data/track_artist_names.txt',sep = '<SEP>',header=None)
        
        # drop unnecessary column and rename columns
        names.drop(columns = [0], inplace = True)
        names.rename(columns={1: "track_id", 2: "artist_name", 3: "song_title"},inplace=True)
        
        # Merge on track id to add new columns
        df = pd.merge(df, names, on = 'track_id',how='left')
        
        return df

    def rec_group_playlist(self,strategy='lm'):
        rankings = self.rankings
        
        rank_cols = [col for col in rankings.columns if 'rank' in col]

        if strategy == 'mp':
             # Most pleasure strategy
            rankings['best_rnk'] = rankings[rank_cols].min(axis=1)
            strat_col = 'best_rnk'
        elif strategy == 'avg':
            # Average rank strategy
            rankings['avg_rnk'] = rankings[rank_cols].mean(axis=1)
            strat_col = 'avg_rnk'
        else:
            # Least misery strategy
            rankings['worst_rnk'] = rankings[rank_cols].max(axis=1)
            strat_col = 'worst_rnk' # default is least misery strategy
        
        top_songs = rankings.sort_values(strat_col)[:self.num_songs]
        top_songs = self.track_artist_names(top_songs)
        
        # Rearrange dataframe to be presentable
        top_songs.insert(0, 'artist_name', top_songs.pop('artist_name'))
        top_songs.insert(1, 'song_title', top_songs.pop('song_title'))
        
        top_songs.drop(columns=['track_id'],inplace=True)
        rating_cols = [col for col in rankings.columns if 'rating' in col]
        top_songs.drop(columns=rating_cols,inplace=True)
        
        char_lim = 30
        top_songs['artist_name'] = top_songs['artist_name'].transform(lambda x: 
                    x[:char_lim] + '...' 
                    if len(x) > char_lim+1 else x)
        top_songs['song_title'] = top_songs['song_title'].transform(lambda x: 
                    x[:char_lim] + '...' 
                    if len(x) > char_lim+1 else x)

        print(top_songs)
        
        return top_songs
    
if __name__ == '__main__':
    # Initial parameters
    user_ids = ['d1ca8b3e78811238cf94ee7caa1868d7ae9e908a',
            '621659a10f52dc4f8b50f205ab85b6d6b7d1b0dc',
            'fef771ab021c200187a419f5e55311390f850a50']
    num_songs = 5
    train_path = 'data/train.csv'
    strategy = 'mp'
    save_path = 'data/results/rankings_svd.csv'
    
    reco = GroupRecommender(user_ids,num_songs)
    
    reco.score_for_users(train_path)
    
    reco.impute_knowns()

    reco.create_rankings()
    
    top_songs = reco.rec_group_playlist(strategy)
    
    top_songs.to_csv(save_path,index='False')
    