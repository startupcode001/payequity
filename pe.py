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

from PE_Functions import *
from PE_Parameter import *

from pathlib import Path

import xlsxwriter
from io import BytesIO

# from streamlit_option_menu import option_menu

# Set Path
st.set_page_config(layout="wide")
# demo_path = Path(__file__).parents[0].__str__()+'/Data/Pay Equity Demo.xlsx'
file_path = Path(__file__).parents[0].__str__()+'/Data/Pay Equity Data Template.xlsx'
display_path = Path(__file__).parents[0].__str__()+'/Data/Display Name.xlsx'
style_path = Path(__file__).parents[0].__str__()+'/Style/style.css'
with open(style_path) as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Set sidebar size
st.markdown(
     """
     <style>
     [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
         width: 260px;
       }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
           width: 260px;
           margin-left: -260px;
       }
    </style>
    """,unsafe_allow_html=True)


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

# Set Functions

# UI *********************************
# Setup Session State:
# if 'demo_run' not in st.session_state:
#     st.session_state['demo_run'] = 'no'
# Side Panel


# m_col1,m_col2 = st.sidebar.columns((1, 1))
# m_col1_but = m_col1.button('See Demo')
# m_col2_but = m_col2.button('Close Demo')

# st.sidebar.markdown("""---""")

# if "demo_box" not in st.session_state:
#     st.session_state.demo_box = False

st.sidebar.header(' 🎯 Start here')
demo_check = st.sidebar.checkbox('See Demo', key='demo_box')

# Step 2: Upload File
if demo_check==False:
    st.sidebar.markdown("Step 1: 🖱️ 'Save link as...'")
    st.sidebar.markdown(get_binary_file_downloader_html(file_path, 'Instruction and Template'), unsafe_allow_html=True)
    uploaded_file = st.sidebar.file_uploader('Step 2: Upload Data Template', type=['xlsx'])
else:
    st.sidebar.caption('Clear the box to launch a new run.')
    uploaded_file = None

st.sidebar.markdown("""---""")

# Main Panel-------------------------------------------
c1, c2 = st.columns((2, 1))

c1.title('PayX')
c1.write('PayX measure the value and the statistical significance of the **net gender/ethnicity pay gap**. That is, we compare pay between men and women, between white and black with similar level, function, location, experience and performance, etc to ensure the difference is gender/ethnicity based. Statistical significance allows us to quantify if a gap is due to chance or gender/ethnicity bias.')
c2.image('Picture/salary.jpeg',use_column_width='auto')
    
main_page = st.container()
with main_page.container():
    main_page_info = main_page.empty()
    # st.write(submit_butt)
    if uploaded_file is not None:
        main_page_info.info('Running input file.')
        analysis(df_submit = uploaded_file, run_demo = False, file_path = file_path, display_path = display_path, main_page = main_page, main_page_info = main_page_info)
        # st.session_state["demo_box"] = False
    else:
        m_info = main_page_info.info('Awaiting the upload of the data template.')
        if demo_check:
            analysis(df_submit = None, run_demo = True, file_path = file_path, display_path = display_path, main_page = main_page, main_page_info = main_page_info)
