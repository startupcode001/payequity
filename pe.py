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
demo_path = Path(__file__).parents[0].__str__()+'/Data/template.xlsx'

# Function
@st.experimental_memo
# Download Excel Template
def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    # href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{bin_file}">{file_label}</a>'
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

@st.experimental_memo
# Run Demo File
def run_demo(demo_path):
    df_demo = pd.read_excel(demo_path,sheet_name="Submission")
    df, df_org, message, exclude_col, r2_raw, female_coff_raw, female_pvalue_raw, r2, female_coff, female_pvalue, fig_pie, before_clean_record, after_clean_record,hc_female = run(df_demo)
    return df, df_org, message, exclude_col, r2_raw, female_coff_raw, female_pvalue_raw, r2, female_coff, female_pvalue, fig_pie,before_clean_record, after_clean_record,hc_female
df, df_org, message, exclude_col, r2_raw, female_coff_raw, female_pvalue_raw, r2, female_coff, female_pvalue, fig_pie,before_clean_record, after_clean_record,hc_female = run_demo(demo_path)

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')
demo_validation = convert_df(df_org)

# UI *********************************

# 'Hello :sunglasses: :heart: '

# Side Panel
st.sidebar.header('Start here')
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
        m_col1,m_col2 = main_page.columns((1, 1))
        m_col1_but = m_col1.button('See a demo')
        m_col2_but = m_col2.button('Close a demo')
        
        if m_col1_but:
            # main_page_info.empty()
            m_info = main_page_info.success('View Demo: '+message.loc[['OVERVIEW']][0])
            main_page.markdown("""---""")
            m_col1_but_col1, m_col1_but_col2, m_col1_but_col3, m_col1_but_col4 = main_page.columns((2, 2, 2, 1))
            
            # Display headcount, Successful Run, Female Percent, download validation file
            m_col1_but_col1.metric('💬 Submission Record',before_clean_record)
            m_col1_but_col2.metric('🏆 Successful Run',after_clean_record)
            m_col1_but_col3.metric('👩 Female %',round(hc_female/after_clean_record,3)*100)
            m_col1_but_col4.download_button('📥 Download Validation as CSV', data=demo_validation, file_name='Data Validation.csv')
            
            # Show R2, Raw Gap, Net Gap
            metric_1, metric_2, metric_3 = main_page.columns((1, 1, 1))            
            metric_1.markdown("<h1 style='text-align: center; color: lightblue;'>Model Robustness</h1>", unsafe_allow_html=True)
            metric_1.pyplot(fig_pie, use_container_width=True)

            metric_2.markdown("<h1 style='text-align: center; color: white; background-color: Green'>Market Benchmark</h1>", unsafe_allow_html=True)
            metric_2.write('In general, model robusiness should be greater than 70%')
            
            metric_3.markdown("<h1 style='text-align: center; color: white; background-color: Green';font-size: 1em>Observation</h1>", unsafe_allow_html=True)
            
        if m_col2_but:
            main_page.empty()