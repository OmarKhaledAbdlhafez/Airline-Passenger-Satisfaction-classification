import streamlit as st
import pandas as pd 
import numpy as np 
import pickle
import matplotlib.pyplot as plt
import seaborn as sns 

st.write('Hello world!')

df = pd.read_csv('train.csv')
df.drop(['Unnamed: 0', 'id'], axis = 1 , inplace = True )
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

sidebar = st.sidebar.selectbox('select page ' , ('overview' , 'EDA' , 'prediction'))

if sidebar == 'overview':
    num_col = df.select_dtypes(exclude = 'O').columns.tolist()
    cat_col = df.select_dtypes(include = 'O').columns.tolist()
    cat = st.selectbox('catorigcal features ' , cat_col)
    fig = plt.figure(figsize=(10, 4))
    sns.countplot(data = df , x= cat)
    st.pyplot(fig)
    num = st.selectbox('numerical features ' , num_col)
    fig = plt.figure(figsize=(10, 4))
    sns.histplot(data = df , x= num)
    st.pyplot(fig)
    

if sidebar == 'EDA':
    
    f, ax = plt.subplots(1, 2, figsize = (15,5))
    sns.boxplot(x = "Customer Type", y = "Age", data = df, ax = ax[0])
    sns.histplot(df, x = "Age", hue = "Customer Type", ax = ax[1])
    st.pyplot(f)

    f, ax = plt.subplots(1, 2, figsize = (15,5))
    sns.boxplot(x = "Class", y = "Age", data = df, ax = ax[0])
    sns.histplot(df, x = "Age", hue = "Class", ax = ax[1])
    st.pyplot(f)


    f, ax = plt.subplots(1, 2, figsize = (15,5))
    sns.boxplot(x = "Class", y = "Flight Distance" , data = df , ax = ax[0])
    sns.histplot(df, x = "Flight Distance", hue = "Class", ax = ax[1])
    st.pyplot(f)

    
    f, ax = plt.subplots(4, 1, figsize = (15,30) )
    sns.countplot(x = 'Class', hue = 'satisfaction', data = df , ax= ax[0])
    sns.countplot(x = 'Inflight wifi service', hue = 'satisfaction', data = df, ax = ax[1])
    sns.countplot(x = 'Seat comfort', hue = 'satisfaction', data = df, ax= ax[2])
    sns.countplot(x = 'Leg room service', hue = 'satisfaction', data = df, ax= ax[3])
    st.pyplot(f)



if sidebar == 'prediction' :
    rate  = [1,2,3,4,5]
    gender = st.selectbox('gender' , df['Gender'].unique().tolist())
    cust_type = st.selectbox ('customer type ' , df['Customer Type'].unique().tolist())
    age = st.number_input('age')
    Type_of_Travel = st.selectbox ('type of travel ' , df['Type of Travel'].unique().tolist())
    Class = st.selectbox ('class' , df['Class'].unique().tolist())
    Flight_Distance = st.number_input('Flight Distance')
    Inflight_wifi_service = st.select_slider('Select a Inflight_wifi_service',options= rate)
    Departure_Arrival_time_convenient = st.select_slider('Departure/Arrival time convenient',options= rate)
    Ease_of_Online_booking = st.select_slider('Ease of Online booking',options= rate)
    Gate_location = st.select_slider('Gate location' ,options= rate)
    Food_and_drink = st.select_slider('Food and drink',options= rate)
    Online_boarding = st.select_slider('Online boarding' ,options= rate)
    Seat_comfort = st.select_slider('Seat comfort',options= rate)
    Inflight_entertainment = st.select_slider('Inflight entertainment',options= rate)
    Onboard_service = st.select_slider('On-board service',options= rate)
    Leg_room_service = st.select_slider('Leg room service',options= rate)
    Baggage_handling = st.select_slider('Baggage handling',options= rate)
    Checkin_service = st.select_slider('Checkin service',options= rate)
    Inflight_service = st.select_slider('Inflight service',options= rate)
    Cleanliness = st.select_slider('Cleanliness',options= rate)
    Departure_Delay_in_Minutes = st.number_input('Departure Delay in Minutes')
    data  = {'Gender' :  gender ,
    'Customer Type' :cust_type ,
    'Age':age ,
    'Type of Travel' :Type_of_Travel ,
    'Class' :Class,
    'Flight Distance' : Flight_Distance,
    'Inflight wifi service' : Inflight_wifi_service,
    'Departure/Arrival time convenient' : Departure_Arrival_time_convenient,
    'Ease of Online booking' : Ease_of_Online_booking,
    'Gate location': Gate_location ,
    'Food and drink' : Food_and_drink,
    'Online boarding' : Online_boarding,
    'Seat comfort' : Seat_comfort,
    'Inflight entertainment' : Inflight_entertainment,
    'On-board service' : Onboard_service,
    'Leg room service' : Leg_room_service,
    'Baggage handling' : Baggage_handling,
    'Checkin service' : Checkin_service,
    'Inflight service' : Inflight_service,
    'Cleanliness' : Cleanliness,
    'Departure Delay in Minutes' :Departure_Delay_in_Minutes 
    }
    t = pd.DataFrame(data = data , index = [0])
    new_data_preprocessed = preprocessor.transform(t)
    out =  model.predict(new_data_preprocessed)
    if st.button('Predict'):
        st.header('Predicted Class')
        st.write(out)
