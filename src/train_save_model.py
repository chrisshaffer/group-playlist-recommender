from recommender import SongRecommender
import pandas as pd
import pickle

train_path = 'data/train_all.csv'

# Reading TRAIN SET from input file into pandas
train_data = pd.read_csv(train_path)

# Creating an instance of your recommender with the right parameters
reco = SongRecommender('cocluster')

# fits on training data, returns a SongRecommender object
model = reco.fit(train_data)

# save model to pickle file
pickle.dump(model, open( "data/model_cocluster.p", "wb" ) )