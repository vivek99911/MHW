from flask import Flask, request, render_template
import pickle

app = Flask(_name_)

model = pickle.load(open('mental_health_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    prediction = model.predict([data])
    
    if prediction == 1:
        result = "Mental Health Risk Detected"
    else:
        result = "No Mental Health Risk"

    return render_template('index.html', prediction_text=result)

if _name_ == "_main_":
    app.run(debug=True)
