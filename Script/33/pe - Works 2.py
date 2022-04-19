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

import pyrebase

# from PE_Functions import *
# from PE_Parameter import *

from pathlib import Path

import xlsxwriter
from io import BytesIO

from streamlit_option_menu import option_menu

# Set Path
st.set_page_config(layout="wide")
# demo_path = Path(__file__).parents[0].__str__()+'/Data/Pay Equity Demo.xlsx'
file_path = Path(__file__).parents[0].__str__()+'/Data/Pay Equity Data Template.xlsx'
display_path = Path(__file__).parents[0].__str__()+'/Data/Display Name.xlsx'
style_path = Path(__file__).parents[0].__str__()+'/Style/style.css'
with open(style_path) as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Set Styles
# metric = st.markdown("""
#     <style>
#     div.css-12w0qpk.e1tzin5v2
#          {background-color: #EFF8F7
#          }
#     div.css-1ht1j8u.e16fv1kl0
#         {font-size: 15px; 
#         }
#     </style>""", unsafe_allow_html=True)

# info_card = st.markdown("""
#     <style>
#     div.css-21e425.e1tzin5v4 {font-size: 5px}
#     </style>""", unsafe_allow_html=True)

m = st.markdown("""
    <style>
    div.stButton > button:first-child {box-shadow: 0px 0px 0px 2px #3498DB;background-color:#3498DB;border-radius:5px;border:2px solid #3498DB;display:inline-block;cursor:pointer;color:#ffffff;font-family:Arial;font-size:13px;padding:8px 25px;text-decoration:none;
    &:active {position:relative;top:1px;}}
    </style>""", unsafe_allow_html=True)

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)


def happy(t):
    st.write(t)

def upload_reset(t):
    st.session_state.upload_count = st.session_state.upload_count + 1
    # st.session_state.read_upload = False
    st.write(t)

if "upload_read" not in st.session_state:
    st.session_state.read_upload = False
    
if "upload_count" not in st.session_state:
    st.session_state.upload_count = 0

st.write(st.session_state.upload_count)
    
st.sidebar.header(' 🎯 Start here')

# Step 1: Download Template
st.sidebar.markdown("Step 1: 🖱️ 'Save link as...'")
# st.sidebar.markdown(get_binary_file_downloader_html(file_path, 'Download Instruction and Data Template'), unsafe_allow_html=True)

# Step 2: Upload File
# reset_but = st.button("Reset and upload a new file")
# if reset_but:
#     st.session_state.read_upload = False
#     st.experimental_rerun()

# uploaded_file = st.sidebar.file_uploader('Step 2: Upload Data Template', type=['xlsx'], on_change=upload_reset,args={'reset done'})

uploaded_file = st.sidebar.file_uploader('Step 2: Upload Data Template', type=['xlsx'])

if (uploaded_file is not None):
    # st.session_state.read_upload = True
    df_submit = pd.read_excel(uploaded_file,sheet_name="Submission")
    col = df_submit.columns.tolist()

    # Step 3: Check empty columns
    st.sidebar.write('Step 3: Review the output in the main panel')
    st.sidebar.write('Step 3: Confirm Selected Configuration')

    config = st.form("Configuration")
    col_select = []
    with config:
        # config.write("A. Choose fair pay confidence internal")
        ci = config.slider(label = 'A. Choose fair pay confidence internal %', min_value = 70, max_value = 99, step = 1, help='Setting at 95% means I want to have a pay range so that 95% of the time, the predicted pay falls inside.')
        col_select = config.multiselect(label = 'C: Select Optional Pay Factors',options=col,default=col,disabled=False)
        submitted_form = config.form_submit_button("🚀 Confirm to Run Analysis'")

    if submitted_form:
        st.write('your submission is: ')
        st.write(col_select)
        st.write(ci)
else:
    st.write('Please upload your file')
    

