from flask import Flask, request, render_template
import numpy as np
import pickle

# Load the pickled model and scalers
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('standscaler.pkl', 'rb') as standscaler_file:
    standscaler = pickle.load(standscaler_file)

with open('minmaxscaler.pkl', 'rb') as minmaxscaler_file:
    minmaxscaler = pickle.load(minmaxscaler_file)

# Create the Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")



@app.route("/predict", methods=['POST'])
def predict():
    N = int(request.form['Nitrogen'])
    P = int(request.form['Phosporus'])
    K = int(request.form['Potassium'])
    temp = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['Ph'])
    rainfall = float(request.form['Rainfall'])

    

    # Create a feature list with the input data
    feature_list = [N, P, K, temp, humidity, ph, rainfall]

    # Perform scaling on the feature list
    scaled_features = minmaxscaler.transform(standscaler.transform([feature_list]))

    # Make the prediction using the loaded model
    prediction = model.predict(scaled_features)

    crop_dict = {
        1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
        8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
        14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
        19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
    }

    
    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivated right there".format(crop)
        image_url = f"static/{crop.lower()}.jpg"
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
        image_url = None

    results = [] 
    results.append({"result": result, "image_url": image_url})

    return render_template('index.html', results=results)


# python main
if __name__ == "__main__":
    app.run(debug=True)
