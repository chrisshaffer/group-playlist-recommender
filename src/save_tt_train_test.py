from prepare_for_surprise import tt_preprocess

# load preprocessed tt data
df = tt_preprocess()

# Split into train/test data 
train = df.sample(frac=0.8,random_state=0)
test = df.drop(train.index)
test = test.set_index('user_id')

train_path = 'data/train_80_20.csv'
train.to_csv(train_path)

test_path = 'data/test_80_20.csv'
test.to_csv(test_path)