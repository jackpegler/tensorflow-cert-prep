#import libraries
import streamlit as st
import numpy as np
import tensorflow as tf
import PIL
from tensorflow.keras.models import load_model

# assigning label names to the corresponding indexes
labels = {0: 'Bread', 1: 'Dairy product', 2: 'Dessert', 3:'Egg', 4: 'Fried food', 5:'Meat',6:'Noodles-Pasta',7:'Rice', 8:'Seafood',9:'Soup',10: 'Vegetable-Fruit'}

st.title("Classify food images")
st.markdown("Simple image classifier to detect food category. Using transfer learning in Tensorflow")

@st.cache(persist=True)
def process_image(user_image):
    print('processing image')
    #code to open the image
    img= PIL.Image.open(user_image)
    print('opened image')
    #resizing the image to (256,256)
    img = img.resize((256,256))
    print('resized')
    #converting image to array
    img = np.asarray(img, dtype= np.float32)
    print('to array')
    #normalizing the image
    img = img / 255
    print('normalised')
    #reshaping the image in to a 4D array
    img = img.reshape(-1,256,256,3)
    print('reshaped')
    #making prediction of the model
    return img

@st.cache(persist=True)
def predict_category(img):
    #making prediction of the model
    predict = model.predict(img)
    print('predicting')
    #getting the index corresponding to the highest value in the prediction
    predict = np.argmax(predict)
    print('got argmax')

    return predict

# @st.cache(persist=True)
# def load-pretrained_model():
#     with tf.device('/cpu:0'):
#         model = load_model('weights.hdf5')
#     return model
#
# #with st.spinner('loading_model'):
#
# model = load-pretrained_()


# # let user upload their image
user_image = st.file_uploader('Upload Your Image', type=['png', 'jpg', 'jpeg'])


# # TODO: could make so just automatically predicts if image updated?
# if st.button("Make Prediction"):
if user_image is not None:
    # TODO: UNderstnad & fix Picke error when cahcing then can load at start
    # load model
    model = load_model('weights.hdf5')

    # process image
    img = process_image(user_image)
    st.image(img)

    # make prediction & show results
    prediction = predict_category(img)
    st.markdown('This is a photo of {}'.format(labels[prediction]))



# TODO: add content of the environment impact!
