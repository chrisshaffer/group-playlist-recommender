from flask import Flask, render_template, request

from group_rec import GroupRecommender

app = Flask(__name__)

@app.route('/')
def student():
   return render_template('student.html')

@app.route('/recommend', methods = ['POST','GET'])
def recommend():
   if request.method == 'POST':
      user_ids = request.form
      
      num_songs = 5
      train_path = 'data/train.csv'
      strategy = 'avg'

      ensemble = False
      
      reco = GroupRecommender(user_ids,num_songs,ensemble)
      
      reco.score_for_users(train_path)
      
      reco.impute_knowns()

      reco.create_rankings()
      
      top_songs = reco.rec_group_playlist(strategy)
      
    
@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      result = request.form
      return render_template("result.html",result = result)

if __name__ == '__main__':
   app.run(debug = True)