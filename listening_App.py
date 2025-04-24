import streamlit as st
import pandas as pd
import joblib
import time

# Baloons Animation
st.sidebar.title("Welcome to My Streamlit App! 🎈")
# st.write("Click the button to celebrate!")
if st.sidebar.button("Celebrate!"):
    st.sidebar.balloons()

if st.sidebar.button("More Balloons!"):
    for _ in range(3):  # Show balloons 3 times
        st.sidebar.balloons()
        time.sleep(2)  # Add a short delay between balloon bursts

if st.sidebar.button("Party Mode!"):
    st.sidebar.balloons()
    time.sleep(2)
    st.sidebar.snow()  # Adds a snowfall effect after balloons


# Load the model
model = joblib.load('listening_model.pkl')
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

st.title("Podcast Listening Time Prediction")
st.write("This app predicts the listening time of a podcast episode based on various features.")
st.write("Please enter the following details:")

# Input fields
# Genre, Number_of_Ads, Publication_Day, Publication_Time, Episode_Sentiment
st.header("Enter the following details:")
Genre = st.selectbox("Genre", df_train['Genre'].unique())
Number_of_Ads = st.number_input("Number of Ads", min_value=0, max_value=10, value=0)
Publication_Day = st.selectbox("Publication Day", df_train['Publication_Day'].unique())
Publication_Time = st.selectbox("Publication Time", df_train['Publication_Time'].unique())
Episode_Sentiment = st.selectbox("Episode Sentiment", df_train['Episode_Sentiment'].unique())

# Map input values to numeric using the label mapping

label_mapping = { # Map correctly with encoded values
    'True Crime':0, 'Comedy':1, 'Education':2, 'Technology':3, 'Health':4,
    'News':5, 'Music':6, 'Sports':7, 'Business':8, 'Lifestyle':9,

    'Thursday':0, 'Saturday':1, 'Tuesday':2, 'Monday':3, 'Sunday':4, 'Wednesday':5,'Friday':6,

    'Night':0, 'Afternoon':1, 'Evening':2, 'Morning':3,

    'Positive':0, 'Negative':1, 'Neutral':2
    }

Genre = label_mapping[Genre]
Number_of_Ads = Number_of_Ads
Publication_Day = label_mapping[Publication_Day]
Publication_Time = label_mapping[Publication_Time]
Episode_Sentiment = label_mapping[Episode_Sentiment]

# Model Prediction using model ,df_train['Episode_Length_minutes'],df_train['Host_Popularity_percentage'],df_train['Guest_Popularity_percentage']
pred = model.predict([[Genre, Number_of_Ads, Publication_Day, Publication_Time, Episode_Sentiment]])

# Display the prediction result on the main screen
st.header("Prediction Result")
st.write("The predicted listening time is:")
st.write(f"{pred[0]:.2f} minutes")

st.write("Please note that this is a prediction and actual listening time may vary.")
st.write("Thank you for using the Podcast Listening Time Prediction app!")