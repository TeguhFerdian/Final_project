import pickle
import joblib
import pandas as pd
# import numpy as np
import streamlit as st
# import matplotlib
# import scipy
import sklearn
# import seaborn as sns
import os

# model = pickle.load(open('model_rf.sav','rb'))
# model = pickle.load(open('model_rf.sav','rb'))

model = joblib.load('holiday.pkl')

st.set_page_config(layout="wide",page_title="Holiday Package Prediction",page_icon="🎡") # https://www.webfx.com/tools/emoji-cheat-sheet/

# ====== About Dataframe =====
st.title('Final Project Data Science Rakamin Bootcamp Batch 32')

st.header('Holiday Package Prediction')
st.write('"Trips & Travel.Com" company wants to enable and establish a viable business model to expand the customer base. One of the ways to expand the customer base is to introduce a new offering of packages.')

st.markdown("[Holiday_Package_Prediction Dataframe](https://www.kaggle.com/datasets/susant4learning/holiday-package-purchase-prediction)")

data = pd.read_csv('dataset/Travel.csv')
st.dataframe(data)

st.download_button(
    label="Download dataframe as CSV",
    data=data.to_csv(),
    file_name='Travel.csv',
    mime='text/csv',
)


# ====== Our Team ======
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.header('Our Team')

col2,col3,col4,col1,col5,col6,col7,col8 = st.columns(8)

col2.image('images/Edgar.png')
col2.caption('Edgar')

col3.image('images/Teguh.png')
col3.caption('Teguh')

col4.image('images/Jannisah.png')
col4.caption('Jannisah')

col1.image('images/sendhy.png')
col1.caption('Sendhy')

col5.image('images/Vionella.png')
col5.caption('Vionella')

col6.image('images/Faris.png')
col6.caption('Faris')

col7.image('images/Jodhi.png')
col7.caption('Jodhi')

col8.image('images/Nanda.png')
col8.caption('Nanda')


# ======Customers With Passport bar chart=====

df_passport = data.groupby(['Passport','ProdTaken']).agg({'CustomerID' : ['nunique']}).reset_index()
df_passport.columns = ['Passport','ProdTaken','Total Customer']
df_passport1 = df_passport.groupby('Passport').agg({'Total Customer' : ['sum']}).reset_index()
df_passport1.columns = ['Passport','Total Customer per Passport']

df_merge = df_passport.merge(df_passport1, on='Passport')


st.header('Customers With Passport')
st.bar_chart(df_merge, x='Passport', y='Total Customer')

# ======ProductPitched by Customer bar chart=====

st.header('ProductPitched by Customer')

df_prod = data.groupby(['ProdTaken','ProductPitched']).agg({'CustomerID' : ['nunique']}).reset_index()
df_prod.columns = ['ProdTaken','ProductPitched','Total Customer']

st.bar_chart(df_prod, x='ProductPitched', y='Total Customer')



st.sidebar.header('Try Our Model Machine Learning')


Age = st.sidebar.slider('How old are you?', 0, 100, 41)
st.sidebar.write("I'm", Age, "years old")

st.sidebar.write("---")

CityTier = st.sidebar.radio('What is your city tier?', (1, 2, 3),2)
st.sidebar.write("I've been", CityTier, "city tier")

st.sidebar.write("---")

DurationOfPitch = st.sidebar.slider('How long duration of pitch did you ?', 0, 100, 6)
st.sidebar.write("I've been", DurationOfPitch, "minutes")

st.sidebar.write("---")

NumberOfFollowups = st.sidebar.radio('How many followups do you have?', ( 1, 2, 3, 4, 5, 6),2)
st.sidebar.write("I've been", NumberOfFollowups, "followups")

st.sidebar.write("---")

ProductPitched = st.sidebar.radio('What is your product pitched?(0 = Basic, 1 = Standard, 2 = Deluxe, 3 = Super Deluxe, 4 = Premium, 5 = King)', (0,1, 2, 3, 4),2)
st.sidebar.write( ProductPitched, "is I've pitched")

st.sidebar.write("---")

PreferredPropertyStar = st.sidebar.radio('What is your preferred property star?', (1, 2, 3, 4, 5),2)
st.sidebar.write('I prefer', PreferredPropertyStar, 'property star')

st.sidebar.write("---")

NumberOfTrips = st.sidebar.slider('How many trips do you have?', 0, 50, 1)
st.sidebar.write("I've been", NumberOfTrips, "trips")

st.sidebar.write("---")

Passport = st.sidebar.select_slider('Do you have a passport? ( 0 : No, 1 : Yes)', (0,1),1)
st.sidebar.write("I've been", Passport, "with passport")

PitchSatisfactionScore = st.sidebar.select_slider('How satisfied are you with your pitch?', (1, 2, 3, 4, 5),2)
st.sidebar.write("I've been", PitchSatisfactionScore, "with my satisfied")

st.sidebar.write("---")

MonthlyIncome = st.sidebar.slider('What is your monthly income?', 0, 100000, 20993)
st.sidebar.write("I've been", MonthlyIncome, "with my monthly income")

st.sidebar.write("---")



# col_left,col_right = st.columns(2)


# with col_left:
#     Age = st.number_input('How old are you?', 0, 100, 25)
#     st.markdown("_Age of customer_")

# with col_left:
#     CityTier = st.number_input('What is your city tier?',1,3,1)
#     st.write("_City tier depends on the development of a city, population, facilities, and living standards. The categories are ordered i.e. Tier 1 > Tier 2 > Tier 3_")

# with col_left:
#     DurationOfPitch = st.number_input('How long did you pitch?', 0, 100, 10)
#     st.write("_Duration of the pitch by a salesperson to the customer_")

# with col_left:
#     NumberOfFollowups = st.number_input('How many followups do you have?', 1, 6, 1)
#     st.write("_Total number of follow-ups has been done by the salesperson after the sales pitch_")

# with col_left:
#     ProductPitched = st.number_input('What is your product pitched?', 0, 4)
#     st.write("_Product pitched by the salesperson_")

# with col_right:
#     PreferredPropertyStar = st.number_input('What is your preferred property star?', 1, 5)
#     st.write("_Preferred hotel property rating by customer_")

# with col_right:
#     NumberOfTrips = st.number_input('How many trips do you have?', 0, 50, 1)
#     st.write("_Average number of trips in a year by customer_")

# with col_right:
#     Passport = st.number_input('Do you have a passport?', 0, 1)
#     st.write("_The customer has a passport or not (0: No, 1: Yes)_")

# with col_right:
#     PitchSatisfactionScore = st.number_input('How satisfied are you with your pitch?', 1, 5)
#     st.write("_Sales pitch satisfaction score_")

# with col_right:
#     MonthlyIncome = st.number_input('What is your monthly income?', 0, 10000, 10000)
#     st.write("_Gross monthly income of the customer_")


prediction = ' '

if st.sidebar.button('Predict Now'):

    prediction = model.predict([[Age,CityTier, DurationOfPitch, NumberOfFollowups, ProductPitched,PreferredPropertyStar,NumberOfTrips,Passport, PitchSatisfactionScore, MonthlyIncome]])
    st.sidebar.write('Prediksi Product Taken',prediction) 

st.write('---')

st.markdown("***open the sidebar and try our Model Machine Learning***" )