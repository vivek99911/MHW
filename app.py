from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))

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

    prediction = model.predict(data)[0]

    if prediction == 1:
        result = "High Risk of Mental Health Issues"
    else:
        result = "Low Risk of Mental Health Issues"

    return render_template("result.html",prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
