from flask import Flask,render_template, url_for , redirect
from flask import request, send_from_directory
import numpy
from numpy import array
from numpy import argmax
from PIL import Image
import pickle
import tensorflow
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image

#from this import SQLAlchemy
app=Flask(__name__,template_folder='template')



# RELATED TO THE SQL DATABASE
app.config['SECRET_KEY'] = '5891628bb0b13ce0c676dfde280ba245'
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///site.db"

dir_path = os.path.dirname(os.path.realpath(__file__))

UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'

with open('tokenizer.pkl', 'rb') as f:
    data = pickle.load(f)

def extract_features(filename):
    model = VGG16()
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    #print(model.summary())
    image = load_img(filename, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    return feature

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Function for generating descriptions
def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text


# procesing uploaded file and predict it
@app.route('/upload', methods=['POST','GET'])
def upload_file():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        file = request.files['image']
        full_name = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(full_name)
        photo = extract_features(full_name)
  	model= load_model("model_9.h5")
        description= generate_desc(model, data, photo, 34)
        descript= description.split(" ")
        descript= " ".join(descript[1:-1])
        return render_template('predict.html', story= descript, image_file_name= file.filename)
        """except :
            flash("Please select the image first !!", "success")      
            return redirect(url_for("caption"))"""


@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/")

# Home page app
@app.route("/home")
def home():
	return render_template("home.html")

# About page
@app.route("/about")
def about():
	return render_template("about.html")

# Cation generator
@app.route("/caption")
def caption():
    return render_template("index.html")

# Intitiating the app        
if __name__ == "__main__":
	app.run(debug=True)
