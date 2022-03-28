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
from PE_Parameter import *

from pathlib import Path

import xlsxwriter
from io import BytesIO

# Set Path
st.set_page_config(layout="wide")
# demo_path = Path(__file__).parents[0].__str__()+'/Data/Pay Equity Demo.xlsx'
file_path = Path(__file__).parents[0].__str__()+'/Data/Pay Equity Data Template.xlsx'
display_path = Path(__file__).parents[0].__str__()+'/Data/Display Name.xlsx'

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

# Set Functions

# UI *********************************
# Setup Session State:
# if 'demo_run' not in st.session_state:
#     st.session_state['demo_run'] = 'no'
# Side Panel

st.sidebar.header(' 🎯 Start here')

st.sidebar.markdown("Step 1: 🖱️ 'Save link as...'")
st.sidebar.markdown(get_binary_file_downloader_html(file_path, 'Download Instruction and Data Template'), unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader('Step 2: Upload Data Template', type=['xlsx'])

submit_butt = False
if uploaded_file is not None:
    submit_butt = st.sidebar.button('🚀 Run Analysis')

st.sidebar.write('Step 3: Review the output in the main panel')

st.sidebar.markdown("""---""")
m_col1,m_col2 = st.sidebar.columns((1, 1))
m_col1_but = m_col1.button('See Demo')
m_col2_but = m_col2.button('Close Demo')

# Main Panel
c1, c2 = st.columns((2, 1))
c2.image('Picture/salary.jpeg',use_column_width='auto')
c1.title('PayX')
c1.write('PayX measure the value and the statistical significance of the **net gender pay gap**. That is, we compare pay between men and women with similar level, function, location, experience and performance, etc to ensure the difference is gender-based. Statistical significance allows us to quantify if a gap is due to chance or gender bias.')

# st.markdown("""---""")

# with st.expander("🔔 See Instruction"):
#     st.write("""To start your analysis, please upload data in sidebar. Check out "See Demo" for a sample output.""")
    # e1, e2 = st.columns((1,4))
    # e1.image('Picture/guide2.jpeg',use_column_width='auto')
    # e2.write("""To start your analysis, please upload data in sidebar. Check out "See Demo" for a sample output.""")

main_page = st.container()
with main_page.container():
    main_page_info = main_page.empty()
    
    if submit_butt == True:
        main_page_info.info('Running input file.')
        analysis(df_submit = uploaded_file, run_demo = False, file_path = file_path, display_path = display_path, main_page = main_page, main_page_info = main_page_info)
        
    else:
        m_info = main_page_info.info('Awaiting the upload of the data template.')
#         m_col1,m_col2,t1 = main_page.columns((1, 1, 2))
        
#         m_col1_but = m_col1.button('See Demo')
#         m_col2_but = m_col2.button('Close Demo')
        if m_col1_but:
            analysis(df_submit = None, run_demo = True, file_path = file_path, display_path = display_path, main_page = main_page, main_page_info = main_page_info)
            
        if m_col2_but:
            main_page.empty()

