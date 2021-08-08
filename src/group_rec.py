import pickle
import pandas as pd
from recommender import SongRecommender
import numpy as np

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

    def rec_group_playlist(self,strategy='lm'):
        rankings = self.rankings
        
        rank_cols = [col for col in rankings.columns if 'rank' in col]

        # Least misery strategy
        rankings['worst_rnk'] = rankings[rank_cols].max(axis=1)

        # Average rank strategy
        rankings['avg_rnk'] = rankings[rank_cols].mean(axis=1)

        # Most pleasure strategy
        rankings['best_rnk'] = rankings[rank_cols].min(axis=1)

        if strategy == 'mp':
            strat_col = 'best_rnk'
        elif strategy == 'avg':
            strat_col = 'avg_rnk'
        else:
            strat_col = 'worst_rnk' # default is least misery strategy
        
        top_songs = rankings.sort_values(strat_col)[:self.num_songs]
        
        print(top_songs)
        
        return top_songs
    
if __name__ == '__main__':
    # Initial parameters
    user_ids = ['d1ca8b3e78811238cf94ee7caa1868d7ae9e908a',
            '621659a10f52dc4f8b50f205ab85b6d6b7d1b0dc']
    num_songs = 5
    train_path = 'data/train_80_20.csv'
    strategy = 'lm'
    save_path = 'data/results/rankings.csv'
    
    reco = GroupRecommender(user_ids,num_songs)
    
    reco.score_for_users(train_path)
    
    reco.impute_knowns()

    reco.create_rankings()
    
    top_songs = reco.rec_group_playlist(strategy)
    
    top_songs.to_csv(save_path)
    