from flask import Flask, request, render_template
import numpy as np
import pickle
import logging

app = Flask(__name__)

# Load the model and scaler
model = pickle.load(open('modelnew.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
        # Extracting form data
        gender = request.form['gender']
        age = float(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        ever_married = request.form['ever_married']
        avg_glucose_level = float(request.form['avg_glucose_level'])
        bmi = float(request.form['bmi'])
        smoking_status = request.form['smoking_status_encoded']

        # Handling categorical variables with safe default values
        gender_mapping = {'Male': 1, 'Female': 0, 'Other': 2}
        married_mapping = {'Yes': 1, 'No': 0}
        smoking_mapping = {'Smoker': 3, 'Non-Smoker': 2, 'Unknown': 0, 'Formerly Smoked':1}

        # Convert categorical variables
        gender = gender_mapping.get(gender)
        ever_married = married_mapping.get(ever_married)
        smoking_status = smoking_mapping.get(smoking_status)


        # Input array for prediction
        input_data = np.array([[gender, age, hypertension, heart_disease, ever_married, avg_glucose_level, bmi, smoking_status]])

        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data_scaled)

        # Return prediction result
        result = "Yes" if prediction[0] == 1 else "No"
        #return f'Predicted chance of stroke: {result}'
        return render_template('result.html',result=result)

    
if __name__ == '__main__':
    app.run(debug=True)