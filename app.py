import numpy as np
import os
import requests
from flask import Flask, request, render_template
import pickle

filename = 'scaler.pkl'

scaler = pickle.load(open(filename, 'rb'))

# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
API_KEY = "ltzFX_cB7r4YXaNJZpZX94ANAXXCGnpYsbo56DhWqg19"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    temp_array = list()

    if request.method == 'POST':

        gre_score = int(request.form["GRE Score"])
        toefl_score = int(request.form["TOEFL Score"])
        university_rating = int(request.form["University Rating"])
        sop = float(request.form["SOP"])
        lor = float(request.form["LOR"])
        cgpa = float(request.form["CGPA"])
        research = request.form["Research"]
        array_of_input_fields = ['greScore', 'toeflScore', 'univRank', 'sop', 'lor', 'cgpa', 'research']
        ar = [gre_score, toefl_score, university_rating, sop, lor, cgpa, research]
        array_of_values_to_be_scored= scaler.transform([ar])
        print(list(array_of_values_to_be_scored[0]))
        payload_scoring = {"input_data": [{"field": [array_of_input_fields], "values": [list(array_of_values_to_be_scored[0])]}]}
        response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/52183faa-6477-4e39-aa8a-2fd0a54375e8/predictions?version=2022-11-17', json=payload_scoring, headers={'Authorization': 'Bearer ' + mltoken})
        print("Scoring response")
        predictions = response_scoring.json()
        xx = (predictions['predictions'][0]['values'][0][1][1])*100
        print(xx)
        print(predictions)
        print(gre_score)
        print(toefl_score)
        print(university_rating)
        print(sop)
        print(lor)
        print(cgpa)


        if xx>50:
            return render_template('chance.html', lower_limit=str(xx)[:5]+"%")
        else:
            return render_template('nochance.html', lower_limit=str(xx)[:5]+"%")

if __name__ == "__main__":
    app.run(debug=True, port=2000)