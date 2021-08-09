from flask import Flask, request, render_template, jsonify
import numpy as np
import pickle
from model import *

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return ''' <p><a href="/predict">Get fraud prediction!</a> </p> 
                   '''

@app.route('/predict', methods=['GET','POST'] )
            
def predict():
    # Load data
    data_path = './data/data_cleaned_labeled.csv'
    X = pd.read_csv(data_path)

    # Drop target column
    X = X.drop(columns=['Fraud'])
    X = X.iloc[:5,:]
    
    # load model
    model_load_path = './src/models/model_xgb.pkl'
    with open(model_load_path, 'rb') as f:
        my_model = pickle.load(f)
        
    # Predict probability of fraud
    predictions = my_model.predict_prob(X)
    event_names = X.event_name.values

    table_str = ''
    for i, p in enumerate(predictions):
        if p > 0.5:
            r = 'High Risk of Fraud'
        elif p > 0.1:
            r = 'Medium Risk of Fraud'
        else:
            r  = 'Low Risk of Fraud'
        table_str = f'ID: {str(event_names[i])}, Probability of Fraud: {str(round(p,3))}, Fraud Risk Level: {r}'

    return table_str

@app.route('/hello', methods=['GET'])
def hello_world():
    return ''' <h1> Hello, World!</h1> '''

@app.route('/form_example', methods=['GET'])
def form_display():
    return ''' <form action="/string_reverse" method="POST">
                <input type="text" name="some_string" />
                <input type="submit" />
               </form>
             '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)