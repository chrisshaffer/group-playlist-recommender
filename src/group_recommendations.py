import pandas as pd
import csv

def single_rec(user_id,pred_path,test_path,save_path,n_recs):
    predictions = pd.read_csv(pred_path)
    test = pd.read_csv(test_path)

    test.drop(columns=['timestamp'],inplace=True)
    test.rename(columns={'rating':'actual_rating'},inplace=True)
    
    df = pd.merge(predictions, test, on=['user_id','track_id']).fillna(1.0)
    user_df = df[df['user_id'] == user_id]

    user_df.sort_values(by=['rating'], ascending = False, inplace=True)
    top_recs = user_df.iloc[:n_recs,:]
    
    # top_recs.to_csv(save_path)
    print(top_recs)
    return top_recs

user_id = 'd1ca8b3e78811238cf94ee7caa1868d7ae9e908a'
pred_path = 'data/results/test_20per_lr01_8020_split.csv'
test_path = 'data/test_80_20.csv'
save_path = 'data/results/test_20per_lr01_8020_split_single_rec.csv'
n_recs = 5
top_recs = single_rec(user_id,pred_path,test_path,save_path,n_recs)