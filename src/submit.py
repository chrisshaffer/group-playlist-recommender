import sys
import logging
import argparse
import pandas as pd
import csv

def compute_score(predictions, actual):
    """Look at 5% of most highly predicted songs for each user.
    Return the average actual rating of those songs.
    """
    actual.drop(columns=['timestamp'],inplace=True)
    # print(predictions)
    # print(actual)
    df = pd.merge(predictions, actual.drop(columns=['rating']), on=['user_id','track_id']).fillna(1.0)
    #df = pd.concat([predictions.fillna(1.0), actual.actualrating], axis=1)

    # for each user
    g = df.groupby(['user_id','track_id']).quantile(.99)
    print(g)
    # # detect the top_5 % as predicted by your algorithm
    # top_5 = g.rating.transform(
    #     lambda x: x >= x.quantile(.95)
    # )

    # return the mean of the actual score on those
    return df.rating[g==1].mean()

    
    
def compute_rmse(predictions, actual):
    # RMSE between predictions and actual ratings

    rmse = ((predictions.rating - actual.rating) ** 2).mean() ** .5
    return rmse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--silent', action='store_true', help="deactivate debug output")
    parser.add_argument('--testing', help="testing set")
    parser.add_argument("predfile", nargs=1, help="prediction file to submit")

    args = parser.parse_args()

    if args.silent:
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p',
                            level=logging.INFO)
    else:
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p',
                            level=logging.DEBUG)
    logger = logging.getLogger('reco-cs')

    path_testing_ = args.testing if args.testing else "data/test.csv"
    logger.debug("using groundtruth from {}".format(path_testing_))

    logger.debug("using predictions from {}".format(args.predfile[0]))
    
    prediction_data = pd.read_csv(args.predfile[0])
    actual_data = pd.read_csv(path_testing_)

    score = compute_score(prediction_data, actual_data)
    rmse = compute_rmse(prediction_data, actual_data)
    print(score)
    logger.debug("score={}".format(score))
    print(rmse)
    logger.debug("rmse={}".format(rmse))
    
    fields=[args.predfile[0],round(score,4),round(rmse,4)]
    with open(r'data/results/results.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
