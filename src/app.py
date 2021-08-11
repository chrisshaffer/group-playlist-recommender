from flask import Flask, render_template, flash, request
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
from group_rec import GroupRecommender

DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = 'SjdnUends821Jsdlkvxh391ksdODnejdDw'

class ReusableForm(Form):
    user1 = TextField('user1:', validators=[validators.required()])
    user2 = TextField('user2:', validators=[validators.required()])
    user3 = TextField('user3:', validators=[validators.required()])

@app.route("/", methods=['GET', 'POST'])
def hello():
    # Generate playlist recommendations based on user ids
    form = ReusableForm(request.form)

    if request.method == 'POST':
        user1=request.form['user1']
        user2=request.form['user2']
        user3=request.form['user3']

        if form.validate():
            flash('Generating playlist...')
            user_ids = [user1,user2,user3]
        
            num_songs = 5
            train_path = 'data/train.csv'
            strategy = 'lm'

            ensemble = False
            
            reco = GroupRecommender(user_ids,num_songs,ensemble)
            
            reco.score_for_users(train_path)
            
            reco.impute_knowns()

            reco.create_rankings()
            
            top_songs = reco.rec_group_playlist(strategy)
            
            top_songs = top_songs.loc[:,['artist_name','song_title']]
            
            top_songs.rename(columns = {'artist_name': 'Artist', 
                                        'song_title':'Song Title'},inplace=True)
            
            top_songs['Track Number'] = range(1,6)
            top_songs.set_index('Track Number', drop=True, inplace=True)
            top_songs.index.name = None
            
            return render_template('view.html',tables=[top_songs.to_html(classes='songs')],titles=[''])
        else:
            flash('Error: All Fields are Required')

    return render_template('index.html', form=form)

if __name__ == "__main__":
    app.run()