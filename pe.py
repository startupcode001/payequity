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

from PE_Functions import *
from pathlib import Path

# Set Path
st.set_page_config(layout="wide")
demo_path = Path(__file__).parents[0].__str__()+'\\Data\\template.xlsx'

# Function
@st.experimental_memo
# Download Excel Template
def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{bin_file}">{file_label}</a>'
    return href

@st.experimental_memo
# Run Demo File
def run_demo(demo_path):
    df_demo = pd.read_excel(demo_path,sheet_name="Submission")
    df, df_org, message, exclude_col, r2_raw, female_coff_raw, female_pvalue_raw, r2, female_coff, female_pvalue, plot_gender = run(df_demo)
    return df, df_org, message, exclude_col, r2_raw, female_coff_raw, female_pvalue_raw, r2, female_coff, female_pvalue, plot_gender
df, df_org, message, exclude_col, r2_raw, female_coff_raw, female_pvalue_raw, r2, female_coff, female_pvalue, plot_gender = run_demo(demo_path)

# st.write(demo_path)
# st.dataframe(data=df_demo.head(4), width=None, height=None)

# UI *********************************
# st.set_page_config(layout="wide")

# Side Panel
st.sidebar.header('Start here')
# st.sidebar.markdown(get_binary_file_downloader_html('Data/template.xlsx', 'Download Input Template'), unsafe_allow_html=True)
st.sidebar.markdown(get_binary_file_downloader_html(demo_path, 'Download Input Template'), unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader('Upload your input Excel file', type=['xlsx'])

# Main Panel
c1, c2 = st.columns((1, 1))
c2.image('Picture/compensation.png',use_column_width='auto')
c1.title('Pay Equity')
c1.write('So what is pay equity? In general, it means compensating employees the same when they perform the same or similar job duties, while accounting for ***pay factors***, such as their job level, job function, experience, performance and tenure with the employer.')

main_page = st.container()
with main_page.container():
    main_page_info = main_page.empty()
    if uploaded_file is not None:
        main_page_info.info('Use input file.')
    else:
        m_info = main_page_info.info('Awaiting the upload of the input file.')
        m1,m2 = main_page.columns((1, 1))
        m1_b = m1.button('See a demo')
        m2_b = m2.button('Close a demo')
        if m1_b:
            # main_page_info.empty()
            m_info = main_page_info.success('View Demo')
            # main_page.
            
            
            main_page.dataframe(data=df.head(4), width=None, height=None)
            main_page.dataframe(data = message)
            main_page.plotly_chart(plot_gender, use_container_width=True)
            
        if m2_b:
            main_page.empty()