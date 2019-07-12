#Import Flask
from flask import Flask, request, jsonify, redirect, send_from_directory
from flask_cors import CORS
#Import python files
import joblib
import requests
import json
import os
from werkzeug.utils import secure_filename
import cv2
from skimage import feature

# On IBM Cloud Cloud Foundry, get the port number from the environment variable PORT
# When running this app on the local machine, default the port to 5000
port = int(os.getenv('PORT', 5000))
print ("Port recognized: ", port)

# Models
model_path = "./model"
model_filename = 'model.joblib'
model_wave = "wave"
model_spiral = "spiral"
encoder_filename = "le.enc"

# Tmp folders for images
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

#Initialize the application service
app = Flask(__name__,static_url_path='')
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
loaded_model_wave = joblib.load(os.path.sep.join([model_path,model_wave,model_filename]))
loaded_model_spiral = joblib.load(os.path.sep.join([model_path,model_spiral,model_filename]))

# Functions
def quantify_image(image):
    # compute the histogram of oriented gradients feature vector for
    # the input image
    features = feature.hog(image, orientations=9,
        pixels_per_cell=(10, 10), cells_per_block=(2, 2),
        transform_sqrt=True, block_norm="L1")

    # return the feature vector
    return features

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict(model_name):
    data = {"success": False}
    if request.method == "POST":
        # check if the post request has the file part
        if 'file' not in request.files:
            print('No file part')
        file = request.files['file']
        # if user does not select file, browser also submit a empty part without filename
        if file.filename == '':
            print('No selected file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            #loaging images
            filename = os.path.join(app.config['UPLOAD_FOLDER'], filename);
            image = cv2.imread(filename)
            output = image.copy()
            output = cv2.resize(output, (128, 128))

            # pre-process the image in the same manner we did earlier
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (200, 200))
            image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

            # quantify the image and make predictions based on the extracted
            # features using the last trained Random Forest
            features = quantify_image(image)

            #loaded model
            loaded_model = joblib.load(os.path.sep.join([model_path,model_name,model_filename]))
            preds = loaded_model.predict([features])

            # load LabelEncoder
            le_loaded = joblib.load(os.path.sep.join([model_path,encoder_filename]))
            label = le_loaded.inverse_transform(preds)[0]
            print (label)

            #Results as Json
            data["predictions"] = []
            r = {"label": label, "score": float(preds[0])}
            data["predictions"].append(r)

            #Success
            data["success"] = True

            return jsonify(data)


#Define a route
@app.route('/')
def default():
    return send_from_directory('client','index.html')

# Wave
@app.route('/wave/predict/',methods=['POST'])
def vgg():
    model_name = "wave"
    return (predict(model_name))

# Spiral
@app.route('/spiral/predict/',methods=['POST'])
def inception():
    model_name = "spiral"
    return (predict(model_name))

# Run de application
app.run(host='0.0.0.0',port=port)
