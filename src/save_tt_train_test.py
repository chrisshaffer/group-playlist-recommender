from prepare_for_surprise import tt_preprocess

# load preprocessed tt data
df = tt_preprocess()

# Split into train/validation data 
frac = 0.8
train = df.sample(frac=frac,random_state=0)
test = df.drop(train.index)
test = test.set_index('user_id')

# Save training data
train_path = 'data/train.csv'
train.to_csv(train_path)

# Save testing data
test_path = 'data/test.csv'
test.to_csv(test_path)