from flask import Flask, request, render_template
import numpy as np
import pickle

# Load models
yield_model = pickle.load(open('C:/Users/mohit/Documents/project/cropyieldpredictor/yield_model.pkl', 'rb'))
preprocessor = pickle.load(open('C:/Users/mohit/Documents/project/cropyieldpredictor/preprocessor.pkl', 'rb'))


# Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('main.html', prediction=None)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        State_Name = request.form['State_Name']
        District_Name = request.form['District_Name']
        Year = request.form['Year']
        Crop = request.form['Crop']
        Area = request.form['Area']

        State_Name = State_Name
        District_Name = District_Name
        Year = Year
        Crop = Crop
        Area = Area

        features = np.array([[State_Name, District_Name, Year, Crop, Area]], dtype=object)
        transformed_features = preprocessor.transform(features)
        try:
            prediction = yield_model.predict(transformed_features).reshape(1, -1)
            prediction_value = prediction[0][0] 
        except Exception as e:
            print(f"Prediction error: {e}")
            prediction = None

        return render_template('main.html', prediction=prediction_value)

if __name__ == "__main__":
    app.run(debug=True)
