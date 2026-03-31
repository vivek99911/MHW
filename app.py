from flask import Flask, render_template, request
import pickle
import numpy as np
import webbrowser
from threading import Timer
import subprocess
import sys

try:
    import xgboost
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])

app = Flask(__name__)

try:
    model = pickle.load(open("model.pkl","rb"))
except Exception as e:
    print("Error loading model:", e)
    model = None

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():

    age = int(request.form['age'])
    gender = int(request.form['gender'])
    family_history = int(request.form['family_history'])
    work_interfere = int(request.form['work_interfere'])
    remote_work = int(request.form['remote_work'])
    tech_company = int(request.form['tech_company'])
    benefits = int(request.form['benefits'])
    care_options = int(request.form['care_options'])
    wellness_program = int(request.form['wellness_program'])
    seek_help = int(request.form['seek_help'])
    anonymity = int(request.form['anonymity'])
    leave = int(request.form['leave'])
    coworkers = int(request.form['coworkers'])

    data = np.array([[age,gender,family_history,work_interfere,remote_work,
                      tech_company,benefits,care_options,wellness_program,
                      seek_help,anonymity,leave,coworkers]])

    if model is None:
        return "Model not loaded properly"

    prediction = model.predict(data)[0]
    prob = model.predict_proba(data)[0]

    print("Probability:", prob[1])   

    if prob[1] > 0.85:
        result = "High Risk of Mental Health Issues"
    elif prob[1] > 0.3:
        result = "Medium Risk of Mental Health Issues"
    else:
        result = "Low Risk of Mental Health Issues"

    return render_template("result.html",prediction=result)

def open_browser():
    webbrowser.open("http://127.0.0.1:5000")

if __name__ == "__main__":
    Timer(1, open_browser).start()
    app.run(debug=True)
    
