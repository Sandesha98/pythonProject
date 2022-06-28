import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask,request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)
MODEL_PATH = 'model/model.h5'

# Load your trained model
model = load_model(MODEL_PATH)

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(150, 150))

    # Preprocessing the image
    x = image.img_to_array(img)

    # Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
   
    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    if preds==0:
        preds="Diseased cotton leaf"
    elif preds==1:
        preds="Diseased cotton plant"
    elif preds==2:
        preds="Fresh cotton leaf"
    else:
        preds="Fresh cotton plant"   
    
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None

if __name__ == '__main__':
    app.run(port=5001,debug=True)