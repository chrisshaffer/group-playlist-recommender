  # Group Playlist Recommender
  
  <p align="center">
    Companies like Spotify create personalized recommendations. How can this be extended to groups of people?
    This project uses the <a href="http://millionsongdataset.com/tasteprofile/"><strong>Echo Nest Taste Profile Dataset»</strong></a> of user play histories to create recommended playlists for groups of users.
  </p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About the Project</a>
      <ul>
        <li><a href="#introduction">Introduction</a></li>
      </ul>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
      <ul>
        <li><a href="#repository-structure">Repository Structure</a></li>
      </ul>
    </li>
    <li>
      <a href="#data">Data</a>
    </li>
    <li>
      <a href="#project-details">Project Details</a>
      <ul>
        <li><a href="#eda">EDA</a></li>
          <ul>
            <li><a href="#ratings">Ratings</a></li>
          </ul>
          <ul>
            <li><a href="#sparsity">Sparsity</a></li>
          </ul>
      </ul>
    </li>
    <li>
      <a href="#modeling">Modeling</a>
    </li>
    <li>
      <a href="#results">Results</a>
    </li>
      <ul>
        <li><a href="#rating-predictions">Rating Predictions</a></li>
      </ul>
      <ul>
        <li><a href="#song-rankings">Song Rankings</a></li>
      </ul>
      <ul>
        <li><a href="#group-recommendations">Group Recommendations</a></li>
      </ul>
    <li>
      <a href="#contact">Contact</a>
    </li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About the Project

### Introduction
In many social situations, like parties and road trips, groups of people with different music tastes will listen to music together. It's hard to please everyone, and it's even harder if you don't know what each person likes. This project extends the personalization of music recommendations for an indvidual to multiple people. There are muliple ways to synthesize the preferences of multiple. Should you average their preferences? Should you try to make sure that no one hates the choices at the cost of excluding someone's favorites? This project explores these question by first generating a music profile for each person, and then applying different strategies to combine their preferences into a single playlist.

### Built With
<b>Python 3.9</b>:

ML modules:
* sci-kit learn
* surprise

Data modules
* pandas
* numpy
* csv
* pickle
* tables
* hdf5_getters

Plotting modules
* matplotlib
* seaborn

Web development
* flask
* wtforms

### Repository structure
* img: Figure image files
* src: Python files

A presentation summarizing the data analysis and results can be found [here](https://github.com/chrisshaffer/group-playlist-recommender/blob/main/Group%20Playlist%20Recommender.pdf).

## Data
This project explores the [Echo Nest Taste Profiles](http://millionsongdataset.com/tasteprofile/) subset of the [Million Song Dataset](http://millionsongdataset.com/). A reference paper explaining the dataset is:
Thierry Bertin-Mahieux, Daniel P.W. Ellis, Brian Whitman, and Paul Lamere. 
[The Million Song Dataset](http://www.columbia.edu/~tb2332/Papers/ismir11.pdf). In Proceedings of the 12th International Society
for Music Information Retrieval Conference (ISMIR 2011), 2011.

The Taste Profiles dataset contains:
1,019,318 unique users
384,546 unique songs
48,373,586 entries

The main table is in the format
 user – song ID –  play counts
 ![image](https://github.com/chrisshaffer/group-playlist-recommender/blob/main/img/example_df.png)

The song information table follows the format:
 song id – song title – artist

## Project Details

### EDA 
#### Ratings
The dataset does not have explicit ratings, so I generated implicit ratings inferred from the song play counts. I made the assumption that user would tend to rate songs that they listened to frequently with high ratings. To generate the ratings, I needed to first explore the data. The play counts per user-song pair ranged from 1 to 10,000, which is a large dynamic range. About half of the songs had only 1 play. 

To deal with the large dynamic range, first I took the logarithm (base-10) of the plays. The play count distribution is shown below:
<p align="center">
  <img src="https://github.com/chrisshaffer/group-playlist-recommender/blob/main/img/plays_hist_annot.png" width="1000" />
</p>
Even after this transformation, the distribution was heavily weighted toward zero, and had a very long tail. I found that only 1% of play counts exceeded 24, so I designated this as the ceiling. I normalized the play counts by this new maximum, took the logarithm, and then mapped them to a 1-5 scale for interpretability. See the equation below:
<p align="center">
  <img src="https://github.com/chrisshaffer/group-playlist-recommender/blob/main/img/rating_eq.png?raw=true" width="1000" />
</p>
Here is the distribution of generated ratings:
<p align="center">
  <img src="https://github.com/chrisshaffer/group-playlist-recommender/blob/main/img/rating_hist_annot.png?raw=true" width="1000" />
</p>
INSERT TABLE
#### Sparsity
Most (99.99%) user-song pairs were missing, from the dataset, as the users only listened to a small fraction of the ~400,000 songs. When the sparsity is too high, learning algorithms have difficulty determining the underlying patterns. I simplified the problem by reducing the dataset to the top 250 songs. This does detract from the real world practicality, but could be remedied in the future by adding more complex song and user information to the training data. This reduced the sparcity from 99.99% to 97%, and retained 11% of the dataset.
<p align="center">
  <img src="https://github.com/chrisshaffer/group-playlist-recommender/blob/main/img/track_popularity_annot.png?raw=true" width="1000" />
</p>
INSERT TABLE
## Modeling 
To train and test different models, I split the data into a random 80/20 train/validation split. All of the models I tested were from the [Surprise](http://surpriselib.com/). Python library for recommender systems. A brief description of the each model algorithm is below, and more information can be found [here](https://surprise.readthedocs.io/en/stable/).
* Normal
  * Fits the ratings to a normal distribution and randomly generates ratings from this distribution. This is a baseline model for comparing other models too.
* Global Mean
  * Assigns the global average rating to each prediction. This is another baseline model.
* K-Nearest Neighbors
  * 
NMF
Co-cluster
ALS Base
SVD

Additionally I combined 3 of the best performing models into an ensemble model. I excluded the ALS Base model because it is too similar to the others. I averaged the predictions of the 3 models, giving weight to each model based on their performance. The ensemble model was superior to all other models, but may be too slow without some modification.

## Results
### Rating Predictions
Here are the results of all of the models, using root-mean-square error of predicted versus actual ratings.
<p align="center">
  <img src="https://github.com/chrisshaffer/group-playlist-recommender/blob/main/img/ensemble_performance.png?raw=true" width="1000" />
</p>
INSERT TABLE
Additionally, I optimized the hyperparameters of the SVD model, both for its use as a standalone model and as part of the ensemble model. I varied the learning rate and number of epochs. I also varied the number of factors in the SVD matrix, but it showed little effect. Below is the error as a function of learning rate and number of epochs.
<p align="center">
  <img src="https://github.com/chrisshaffer/group-playlist-recommender/blob/main/img/rmse_svd_hyp.jpg?raw=true" width="1000" />
</p>
The results showed that a low learning rate (0.001) with a high number of epochs (125). Pushing these values to more extreme may offer marginal improvement in performance, but the training time increases nonlinearly, so I stopped there.

### Song Rankings
I trained the model on all of the data in preparation for generating recommendations. The ranking algorithm is as follows. For a user, ratings for all 250 songs are predicted by the model. The known song ratings are imputed into the predictions, constituting ~3% of the predictions. The ratings are then sorted, and assigned a ranking of 0-249, where 0 is the ranking of the highest rated song.

### Group Recommendations
When a group of users is submitted for recommendations, the algorithm generates rankings for each user as described above. I implemented 3 different strategies for synthesizing the sets of rankings into a single group rating. There is some research behind the strategies rooted in psychology. The first, and most basic strategy is to average the rankings of the users for each song. This is referred to as the "average strategy". A playlist recommendation for 5 songs for 3 from the dataset is below.
<p align="center">
  <img src="https://github.com/chrisshaffer/group-playlist-recommender/blob/main/img/average_recs.png?raw=true" width="1000" />
</p>

Sometimes this results in recommendations for songs that most group members rate highly, but one or some rate very low. The "least misery strategy" addresses this issue by defining the group ranking for each song as the worst (maximum) ranking of the group members. The idea is that "the group is only as happy as its least happy member". A playlist recommendation using the "least misery strategy" is below.
<p align="center">
  <img src="https://github.com/chrisshaffer/group-playlist-recommender/blob/main/img/least_misery_recs.png?raw=true" width="1000" />
</p>

Last, the downside of the "least misery strategy" is that it may generate recommendations that no one hates, but none or few of the members love. The complement to the "least misery strategy" is the "most pleasure strategy." Group rankings for each song are defined as the best (minimum) ranking of the group members. The enjoyment of others may be contagious, so group members may feed off of the energy of the happiest member. An example using this strategy is below.
<p align="center">
  <img src="https://github.com/chrisshaffer/group-playlist-recommender/blob/main/img/most_pleasure_recs.png?raw=true" width="1000" />
</p>

=
<!-- Contact -->
## Contact

Author: Christopher Shaffer

[Email](christophermshaffer@gmail.com)

[Github](https://github.com/chrisshaffer)
