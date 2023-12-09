
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image

# Set page title and background color
st.set_page_config(page_title='Diabetes Mellitus Classification', page_icon=':pill:', layout='wide', initial_sidebar_state='auto')

# Set custom CSS styles
st.markdown("""
<style>
    .title {
        text-align: center;
        font-size: 32px;
        margin-bottom: 30px;
    }
    
    .image {
        display: flex;
        justify-content: center;
        margin-bottom: 50px;
    }
    
    .input-section {
        display: flex;
        justify-content: space-between;
        margin-bottom: 20px;
    }
    
    .prediction {
        text-align: center;
        font-size: 24px;
        margin-top: 30px;
    }
</style>
""", unsafe_allow_html=True)

# Create navigation sidebar
nav_selection = st.sidebar.radio("Navigation", ["Home", "About Diabetes"])

# Home page
if nav_selection == "Home":
    st.title('Diabetes Mellitus Classification')
    # Rest of your code for the home page
    
# About Diabetes page
elif nav_selection == "About Diabetes":
    st.title('About Diabetes Mellitus')
    
    img = Image.open('images.jfif')
    img = img.resize((700, 200))
    st.image(img, use_column_width=True)

    st.write("Diabetes mellitus is a chronic metabolic disorder characterized by high blood sugar levels. It occurs when the body either doesn't produce enough insulin or can't effectively use the insulin it produces.")
    
    st.write("There are several types of diabetes, including type 1 diabetes, type 2 diabetes, and gestational diabetes. Type 1 diabetes is an autoimmune condition where the body's immune system mistakenly attacks and destroys the insulin-producing cells in the pancreas. Type 2 diabetes is more common and usually develops due to a combination of genetic and lifestyle factors.")
    
    st.write("If left unmanaged, diabetes can lead to various complications such as heart disease, kidney damage, nerve damage, and vision problems. Early detection and proper management are crucial in controlling diabetes and preventing complications.")
    
    # Add more information or resources as needed
    
    st.write("For more detailed information about diabetes, you can refer to reputable sources such as the American Diabetes Association (ADA) or consult with a healthcare professional.")
    
     
    # Hide the input section
    st.stop()
# Display title and image
#st.markdown("<h1 class='title'>Diabetes Mellitus Classification</h1>", unsafe_allow_html=True)
img = Image.open('image1.png')
img = img.resize((700, 200))
st.image(img, use_column_width=True)

# Load model and scaler
model = pickle.load(open('RF_class_model.pkl', 'rb'))
scaler = pickle.load(open('scal.pkl', 'rb'))
encoder = pickle.load(open('enc.pkl', 'rb'))

# Function to predict diabetes
def predict():
    # Input section layout
    c1, c2 = st.columns(2)
    with c1:
        Age = st.number_input('Please enter your age',step=1)
        Gender = st.selectbox('Gender', ['Female', 'Male'])
        Polyuria = st.selectbox('Do you experience excessive urination?', ['No', 'Yes'])
        Polydipsia = st.selectbox('Do you experience excessive thirst?', ['No', 'Yes'])
        sudden_weight_loss = st.selectbox('Do you experience sudden weight loss?', ['No', 'Yes'])
        weakness = st.selectbox('Do you experience general body weakness?', ['No', 'Yes'])
        Polyphagia = st.selectbox('Do you experience excessive hunger?', ['No', 'Yes'])
        Genital_thrush = st.selectbox('Do you suffer genital infections?', ['No', 'Yes'])

    with c2:
        visual_blurring = st.selectbox('Do you experience blurred vision?', ['No', 'Yes'])
        Itching = st.selectbox('Do you experience body itching?', ['No', 'Yes'])
        Irritability = st.selectbox('Do you experience nausea?', ['No', 'Yes'])
        delayed_healing = st.selectbox('Do you suffer from delayed healing?', ['No', 'Yes'])
        partial_paresis = st.selectbox('Do you experience weakened muscle movement?', ['No', 'Yes'])
        muscle_stiffness = st.selectbox('Do you suffer from muscle stiffness?', ['No', 'Yes'])
        Alopecia = st.selectbox('Do you experience sudden hair loss?', ['No', 'Yes'])
        Obesity = st.selectbox('Do you suffer from obesity?', ['No', 'Yes'])

    feat = np.array([Age, Gender, Polyuria, Polydipsia, sudden_weight_loss,
                     weakness, Polyphagia, Genital_thrush, visual_blurring,
                     Itching, Irritability, delayed_healing, partial_paresis,
                     muscle_stiffness, Alopecia, Obesity]).reshape(1, -1)
    cols = ['Age', 'Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss',
            'weakness', 'Polyphagia', 'Genital thrush', 'visual blurring',
            'Itching', 'Irritability', 'delayed healing', 'partial paresis',
            'muscle stiffness', 'Alopecia', 'Obesity']
    feat1 = pd.DataFrame(feat, columns=cols)

    return feat1

frame = predict()

def prepare(df):
    enc_data = pd.DataFrame(encoder.transform(df[['Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss',
                                                    'weakness', 'Polyphagia', 'Genital thrush', 'visual blurring',
                                                    'Itching', 'Irritability', 'delayed healing', 'partial paresis',
                                                    'muscle stiffness', 'Alopecia', 'Obesity']]).toarray())
    enc_data.columns = encoder.get_feature_names(['Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss',
                                                  'weakness', 'Polyphagia', 'Genital thrush', 'visual blurring',
                                                  'Itching', 'Irritability', 'delayed healing', 'partial paresis',
                                                  'muscle stiffness', 'Alopecia', 'Obesity'])
    df = df.join(enc_data)

    df.drop(['Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss',
             'weakness', 'Polyphagia', 'Genital thrush', 'visual blurring',
             'Itching', 'Irritability', 'delayed healing', 'partial paresis',
             'muscle stiffness', 'Alopecia', 'Obesity'], axis=1, inplace=True)

    cols = df.columns
    df = scaler.transform(df)
    df = pd.DataFrame(df, columns=cols)

    return df

frame2 = prepare(frame)

if st.button('Predict'):
    frame2 = prepare(frame)
    pred = model.predict(frame2)
    if pred[0] == 0:
        st.markdown("<p class='prediction'>This individual does not have diabetes</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='prediction'>This individual has diabetes</p>", unsafe_allow_html=True)
