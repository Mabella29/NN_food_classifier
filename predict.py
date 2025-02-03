from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from flask import Flask, request, jsonify

# Load the trained model
model = load_model('models/vgg19_food_classifier.h5')

# Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img = image.load_img(file, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    
    return jsonify({'class': int(predicted_class)})

if __name__ == '__main__':
    app.run(debug=True)