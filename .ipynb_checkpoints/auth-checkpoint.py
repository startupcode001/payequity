import streamlit as st
import pandas as pd
import numpy as np

from patsy import dmatrices
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from statsmodels.sandbox.regression.predstd import wls_prediction_std

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import os

import locale

import pyrebase as pb

from PE_Functions import *
from PE_Parameter import *

from pathlib import Path

import xlsxwriter
from io import BytesIO

from datetime import datetime

# Configuration Key
firebaseConfig = {
    'apiKey': "AIzaSyBlDNmyf4KhVwXaOYC6D0I0KR03XYQ78yU",
    'authDomain': "pay-equity.firebaseapp.com",
    'projectId': "pay-equity",
    'storageBucket': "pay-equity.appspot.com",
    'messagingSenderId': "911725548540",
    'appId': "1:911725548540:web:a0aebc04538539ea941879",
   ' measurementId': "G-05TJ6HSVY4",
    'databaseURL': "https://pay-equity-default-rtdb.firebaseio.com/"
}

# Firebase Authentication
fb = pb.initialize_app(firebaseConfig)
auth = fb.auth()

# Database
db = fb.database()
storage = fb.storage()
st.sidebar.title("Our community app")

# Authentication
choice = st.sidebar.selectbox('login/Signup', ['Login', 'Sign up'])

# Obtain User Input for email and password
email = st.sidebar.text_input('Please enter your email address')
password = st.sidebar.text_input('Please enter your password',type = 'password')

# App 

# Sign up Block
if choice == 'Sign up':
    username = st.sidebar.text_input(
        'Please input your user name', value='Default')
    company = st.sidebar.text_input(
        'Please input your company name', value='Default')
    submit = st.sidebar.button('Create my account')

    if submit:
        user = auth.create_user_with_email_and_password(email, password)
        st.success('Your account is created suceesfully!')
        st.balloons()
        # Sign in
        user = auth.sign_in_with_email_and_password(email, password)
        db.child(user['localId']).child("ID").set(user['localId'])
        db.child(user['localId']).child("Username").set(username)
        db.child(user['localId']).child("Company").set(company)
        db.child(user['localId']).child("Email").set(email)
        db.child(user['localId']).child("Password").set(password)
        
        st.title('Welcome' + username)
        st.info('Login via login drop down selection')

# Login Block
if choice == 'Login':
    login = st.sidebar.checkbox('Login')
    if login:
        try:
            user = auth.sign_in_with_email_and_password(email,password)
            # user_info = auth.get_account_info()
        except:
            st.write('User not found, please sign up with drop down selection')
            st.stop()
           
        st.title('Welcome')
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        bio = st.radio('Jump to',['Home','Calculation'])
        
        if bio == 'Calculation':
            cal_start = st.number_input('Insert a number')
            st.write('your starting input is: '+str(cal_start))
            
            cal_add = st.number_input('Add a number')
            st.write('your add number is: '+ str(cal_add))
            
            cal_result = cal_start+cal_add
            st.write('your final number is: '+str(cal_result))
            
            butt_save = st.button('Save result')
            if butt_save:
                now = datetime.now()
                dt_string = now.strftime("%d/%m/%Y %H:%M:%S")              
                cal_save = {'Calculation' : cal_result,
                        'Timestamp' : dt_string} 
                
                db.child(user['localId']).child("Calculation").push(cal_save)
                st.balloons()
