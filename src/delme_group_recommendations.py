import pandas as pd
import warnings

warnings.filterwarnings("ignore")

def track_artist_names(df):
    """Extract song_title and artist_name for tracks in given DataFrame

    Args:
        df (pandas DataFrame): DataFrame of top recommended songs

    Returns:
        pandas DataFrame: Initial DataFrame with additional song_title, artist_name columns
    """    
    
    # make actual_rating index into a column
    df.reset_index(inplace=True)
    
    # read in map of track_id to artist_name and song_title
    names = pd.read_csv('data/track_artist_names.txt',sep = '<SEP>',header=None)
    
    # drop unnecessary column and rename columns
    names.drop(columns = [0], inplace = True)
    names.rename(columns={1: "track_id", 2: "artist_name", 3: "song_title"},inplace=True)
    
    # Merge on track id to add new columns
    df = pd.merge(df, names, on = 'track_id',how='left')
    
    return df

def single_rec(user_id,pred_path,test_path,save_path,n_recs):
    """Return a pandas DataFrame for top n song recommendations for a user

    Args:
        user_id (str): user_id to return recommendations for
        pred_path (str): file path to predicted ratings
        test_path (str): file path to actual ratings
        save_path (str): save path for recommendations
        n_recs (int): number of recommendations

    Returns:
        pandas DataFrame: columns are actual_ranking, track_id, rating,
            actual_rating, artist_name, song_title
    """    
    predictions = pd.read_csv(pred_path)
    test = pd.read_csv(test_path)

    test.drop(columns=['timestamp'],inplace=True)
    test.rename(columns={'rating':'actual_rating'},inplace=True)
    
    df = pd.merge(predictions, test, on=['user_id','track_id']).fillna(1.0)
    user_df = df[df['user_id'] == user_id]

    user_df = user_df.sort_values(by='actual_rating',ascending=False).reset_index().drop(columns=['index'])
    user_df.index.names = ['actual_ranking(of ' + str(user_df.shape[0]) + ')']
    user_df.sort_values(by=['rating'], ascending = False, inplace=True)
    top_recs = user_df.iloc[:n_recs,:]
    
    top_recs = track_artist_names(top_recs)
    
    # top_recs.to_csv(save_path)
    pd.set_option('display.max_colwidth', -1)
    print(top_recs.drop(columns='user_id'))
    
    return top_recs

def multiple_recs(user_id_lst,pred_path,test_path,save_path,n_recs):
    for idx, user_id in enumerate(user_id_lst):
        if idx == 0:
            top_recs = single_rec(user_id,pred_path,test_path,save_path,n_recs)
        else:
            new_top_recs = single_rec(user_id,pred_path,test_path,save_path,n_recs)
            top_recs = pd.concat([top_recs,new_top_recs],axis=1,join='inner', keys=['1', '2'])

    pd.set_option('display.max_colwidth', -1)
    print(top_recs.iloc[0,:])
    
    return top_recs

user_id = 'd1ca8b3e78811238cf94ee7caa1868d7ae9e908a'
user_id2 = '621659a10f52dc4f8b50f205ab85b6d6b7d1b0dc'
user_id_lst = [user_id,user_id2]
pred_path = 'data/results/test_20per_lr01_8020_split.csv'
test_path = 'data/test_80_20.csv'
save_path = 'data/results/test_20per_lr01_8020_split_single_rec.csv'
n_recs = 5
top_recs = multiple_recs(user_id_lst,pred_path,test_path,save_path,n_recs)