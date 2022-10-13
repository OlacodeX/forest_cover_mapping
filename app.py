import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

## Create a new instance of the flask app
app = Flask(__name__)
## Load the model
rfcmodel = pickle.load(open('rfc_model.pkl', 'rb'))

## Define the landing page route
@app.route('/')
## Define the landing page resource
def home():
    return render_template('home.html')

## Define route for prediction API for communicating with your app
@app.route('/predict_api', methods=['POST'])
## Define the API resource
def predict_api():
    data=request.json['data']
    print(data)
    new_data = np.array(list(data.values())).reshape(1,-1)
    output = rfcmodel.predict(new_data)
    print(output[0])
    ## this output is returned as an ndarray. To make it possible to jsonify it, I will convert it to a list
    final_output = output.tolist()
    ## Now I just pick the first and obviously only element in the list. I specified the index to remove the square brackets
    return jsonify(final_output[0])

@app.route('/predict', methods=['POST'])
def predict():
    data= [float(x) for x in request.form.values()]
    final_input = np.array(list(data)).reshape(1,-1)
    print(final_input)
    output = rfcmodel.predict(final_input)
    return render_template("result.html", prediction_text="The predicted cover type is {}".format(output[0]))

## Run the app now
if __name__=="__main__":
    app.run(debug=True)