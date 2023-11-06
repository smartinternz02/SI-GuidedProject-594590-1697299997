from flask import Flask, request, render_template
from tensorflow.keras.preprocessing import image
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)

# Define a function to load your trained model
def load_model():
    model_path = 'C:\\Users\\varun\\OneDrive\\Desktop\\Fall Sem\\Smart Internz AI&ML\\Covid Detection\\4.Project Development Phase\\Covid_Flask\\Stacked_Model_1.h5'
    model = keras.models.load_model(model_path)
    return model

model = load_model()

# Define a function to preprocess the input image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/protect')
def protect():
    return render_template('protect.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/doctors')
def doctors():
    return render_template('doctors.html')

@app.route('/news')
def news():
    return render_template('news.html')

@app.route('/predict',methods = ['GET','POST'])
def upload():
    if request.method=='POST':
        f = request.files['image']
        basepath=os.path.dirname(__file__)
        filepath = os.path.join(basepath,'uploads',f.filename)
        f.save(filepath)

        img = image.load_img(filepath,target_size =(224,224))
        x = image.img_to_array(img)/255
        x = np.expand_dims(x,axis = 0)
        pred =np.argmax(model.predict(x),axis=1)
        print(int(pred))
    
        lis=['COVID','LUNG OPACITY','NORMAL','PNEUMONIA']
        # Process the prediction result and format it as needed
        if int(pred)==0:
            covid_prob="Positive"
        else:
            covid_prob="Negative"
    
        lung_disorder_prob = lis[int(pred)]
        
        return render_template('index.html', covid_prob=covid_prob, lung_disorder_prob=lung_disorder_prob)

if __name__ == '__main__':
    app.run(debug=True)



 