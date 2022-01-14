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

from PE_Functions import *
from pathlib import Path

# Set Path
st.set_page_config(layout="wide")
demo_path = Path(__file__).parents[0].__str__()+'/Data/template.xlsx'

# Set Style
# st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

# Set Style
# m = st.markdown("""
#     <style>
#     div.stButton > button:first-child {box-shadow: 0px 0px 0px 2px #9fb4f2;background-color:#3498DB;border-radius:10px;border:1px solid #3498DB;display:inline-block;cursor:pointer;color:#ffffff;font-family:Arial;font-size:13px;padding:12px 37px;text-decoration:none;text-shadow:0px 1px 0px #283966;
#     &:hover {background-color:#476e9e;}
#     &:active {position:relative;top:1px;}}
#     </style>""", unsafe_allow_html=True)

m = st.markdown("""
    <style>
    div.stButton > button:first-child {box-shadow: 0px 0px 0px 2px #3498DB;background-color:#3498DB;border-radius:10px;border:1px solid #3498DB;display:inline-block;cursor:pointer;color:#ffffff;font-family:Arial;font-size:13px;padding:12px 37px;text-decoration:none;
    &:active {position:relative;top:1px;}}
    </style>""", unsafe_allow_html=True)

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

# Set Functions

# UI *********************************
# Setup Session State:
# if 'demo_run' not in st.session_state:
#     st.session_state['demo_run'] = 'no'
# Side Panel

st.sidebar.header(' 🔔 Start here')
st.sidebar.markdown(get_binary_file_downloader_html(demo_path, 'Step 1: Download Instruction and Data Template'), unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader('Step 2: Upload Data Template', type=['xlsx'])

submit_butt = False
if uploaded_file is not None:
    submit_butt = st.sidebar.button('🚀 Run Analysis')

st.sidebar.write('Step 3: Review result in main panel')
st.sidebar.markdown("""---""")

# Main Panel
c1, c2 = st.columns((2, 1))
c2.image('Picture/salary.jpeg',use_column_width='auto')
c1.title('Pay Equity')
c1.write('So what is pay equity? In general, it means compensating employees the same when they perform the same or similar job duties, while accounting for ***pay factors***, such as their job level, job function, experience, performance and tenure with the employer.')

# st.markdown("""---""")

with st.expander("🎯 See Instruction"):
    e1, e2 = st.columns((1,4))
    e1.image('Picture/guide2.jpeg',use_column_width='auto')
    e2.write("""Placeholder for instruction...:) """)

main_page = st.container()
with main_page.container():
    main_page_info = main_page.empty()
    
    if submit_butt == True:
        main_page_info.info('Running input file.')
        analysis(df_submit = uploaded_file, run_demo = False, demo_path = demo_path, main_page = main_page, main_page_info = main_page_info)
        
    else:
        m_info = main_page_info.info('Awaiting the upload of the data template.')
        m_col1,m_col2,t1 = main_page.columns((1, 1, 2))
        
        m_col1_but = m_col1.button('See Demo')
        m_col2_but = m_col2.button('Close Demo')
        
        if m_col1_but:
            analysis(df_submit = None, run_demo = True, demo_path = demo_path, main_page = main_page, main_page_info = main_page_info)
            
        if m_col2_but:
            main_page.empty()

