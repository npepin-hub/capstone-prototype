from flask import Flask, render_template, request, flash, redirect, url_for
from flask_uploads import IMAGES, UploadSet, configure_uploads

from PIL import Image

import os
import sys
import numpy as np

sys.path.append('../src/')
#sys.path.append('../')
from config import settings
import GloVepreprocessing
from predict import predict


sys.path.append('./pachyderm/')
#import pachydermtest
from pachydermtest import get_features_model, get_inference_model


app = Flask(__name__)
photos = UploadSet("photos", IMAGES)
app.config["UPLOADED_PHOTOS_DEST"] = "/var/img"
app.config["SECRET_KEY"] = 12
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png']



preprocessor = GloVepreprocessing.preprocessor_factory()
features_model = get_features_model()
inference_model = get_inference_model("pachyderm/testdata/saved/saved_model.h5", preprocessor)

configure_uploads(app, photos)





@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        url = photos.url(filename)
        image = Image.open(photos.path(filename))
        caption = predict(image, preprocessor, features_model, inference_model)    

        return render_template('upload.html', url=url, caption=caption)
    return render_template('upload.html')
 


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')