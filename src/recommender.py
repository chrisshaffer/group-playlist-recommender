import logging
import numpy as np
import pandas as pd
from surprise import SVD, SlopeOne, NMF, KNNBaseline
from surprise.prediction_algorithms.co_clustering import CoClustering
from surprise.prediction_algorithms.baseline_only import BaselineOnly
from surprise.prediction_algorithms.random_pred import NormalPredictor
from surprise import Dataset
from surprise import Reader
from sklearn.metrics.pairwise import cosine_similarity

class SongRecommender():
    """Basic song recommender system."""

    def __init__(self, model_name='svd'):
        """Constructs a SongRecommender"""
        self.logger = logging.getLogger('reco-cs')
        self.model_name = model_name

    def fit(self, ratings):
        """
        Trains the recommender on a given set of ratings.

        Parameters
        ----------
        ratings : pandas dataframe, shape = (n_ratings, 4)
                  with columns 'user_id', 'track_id', 'likes', 'timestamp'

        Returns
        -------
        self : object
            Returns self.
        """
        self.logger.debug("starting fit")

        # processing for Surprise
        ratings = ratings.sample(frac=1)
        ratings = Dataset.load_from_df(ratings[['user_id', 'track_id', 'rating']],
                                       reader=Reader(rating_scale = (1,5)))
        self.trainset = ratings.build_full_trainset()
        
        if self.model_name == 'svd':
            self.algo = SVD(lr_all=0.001,n_epochs=125)
        elif self.model_name == 'slopeone':
            self.algo = SlopeOne()
        elif self.model_name == 'nmf':
            self.algo = NMF()
        elif self.model_name == 'knnbaseline':
            self.algo = KNNBaseline() 
        elif self.model_name == 'cocluster':
            self.algo = CoClustering() 
        elif self.model_name == 'baseline':
            self.algo = BaselineOnly()
        elif self.model_name == 'normal':
            self.algo = NormalPredictor()            
        self.algo.fit(self.trainset)

        self.logger.debug("finishing fit")
        return(self)

    def transform(self, requests):
        """
        Predicts the ratings for a given set of requests.

        Parameters
        ----------
        requests : pandas dataframe, shape = (n_ratings, 4)
                  with columns 'user_id', 'track_id', 'rating', 'timestamp'

        Returns
        -------
        dataframe : a pandas dataframe with columns 'user_id', 'track_id', 'rating'
                    column 'rating' contains the predicted rating
        """
        self.logger.debug("starting predict")
        self.logger.debug("request count: {}".format(requests.shape[0]))


        testset = Dataset(reader=Reader()).construct_testset(raw_testset = requests.values)
        predictions = self.algo.test(testset)
        
        pred_base = [(pred.uid,pred.iid,pred.est) for pred in predictions]

        predictions = pd.DataFrame(pred_base,columns=['user_id', 'track_id', 'rating'])

        self.logger.debug("finishing predict")
        return(predictions)

    
if __name__ == "__main__":
    logger = logging.getLogger('reco-cs')
    logger.critical('you should use run.py instead')
