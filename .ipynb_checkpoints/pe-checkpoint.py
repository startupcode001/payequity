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
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">{file_label}</a>'
    return href

@st.experimental_memo
# Run Demo File
def run_demo(demo_path):
    df_demo = pd.read_excel(demo_path,sheet_name="Submission")
    df, df_org, message, exclude_col, r2_raw, female_coff_raw, female_pvalue_raw, r2, female_coff, female_pvalue, before_clean_record, after_clean_record,hc_female,fig_r2_gender_gap,fig_raw_gender_gap,fig_net_gender_gap = run(df_demo)
    return df, df_org, message, exclude_col, r2_raw, female_coff_raw, female_pvalue_raw, r2, female_coff, female_pvalue, before_clean_record, after_clean_record,hc_female,fig_r2_gender_gap,fig_raw_gender_gap,fig_net_gender_gap
df, df_org, message, exclude_col, r2_raw, female_coff_raw, female_pvalue_raw, r2, female_coff, female_pvalue, before_clean_record, after_clean_record,hc_female,fig_r2_gender_gap,fig_raw_gender_gap,fig_net_gender_gap = run_demo(demo_path)

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')
demo_validation = convert_df(df_org)


# Navigation Bar

st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

st.markdown("""
<nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #3498DB;">
  <a class="navbar-brand" href="https://youtube.com/dataprofessor" target="_blank">Pay Equity</a>
  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
    <span class="navbar-toggler-icon"></span>
  </button>
  <div class="collapse navbar-collapse" id="navbarNav">
    <ul class="navbar-nav">
      <li class="nav-item active">
        <a class="nav-link disabled" href="#">Home <span class="sr-only">(current)</span></a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="https://youtube.com/dataprofessor" target="_blank">Sign up/Login in</a>
      </li>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="https://twitter.com/thedataprof" target="_blank">About Us</a>
      </li>
    </ul>
  </div>
</nav>
""", unsafe_allow_html=True)

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
            m_col1_but_col4.download_button('📥 Download exclusions', data=demo_validation, file_name='Data Validation.csv')
            
            # Show R2
            metric_R2_1, metric_R2_2, metric_R2_3 = main_page.columns((1, 1, 1))            
            metric_R2_1.markdown("<h1 style='text-align: left; vertical-align: bottom; font-size: 200%; color: #3498DB; opacity: 0.7'>Robustness</h1>", unsafe_allow_html=True)
            metric_R2_1.pyplot(fig_r2_gender_gap, use_container_width=True)
            
            metric_R2_2.markdown("<h1 style='text-align: left; vertical-align: bottom;color: #3498DB; font-size: 200%; opacity: 0.7'>Benchmark</h1>", unsafe_allow_html=True)
            metric_R2_2.markdown("<h1 style='text-align: left; vertical-align: bottom;color: Green; font-size: 150%; opacity: 0.7'> 🌐 70% ~ 100%  </h1>" "  \n"  "Model Robutness measures how well the pay factors drive pay decisions. For example 80% means pay factors explain 80 percent of the pay variation among employees.", unsafe_allow_html=True)
            
            metric_R2_3.markdown("<h1 style='text-align: left; vertical-align: bottom;color: #3498DB; font-size: 200%; opacity: 0.7'>Observation</h1>", unsafe_allow_html=True)
            if r2>=0.7:
                metric_R2_3.markdown("<h1 style='text-align: left; vertical-align: bottom;color: Green; font-size: 150%; opacity: 0.7'> ✔️ Align with market  </h1>" "  \n"  "The higher robustness, the more accurate the model make pay explination and predictions", unsafe_allow_html=True)
            else:
                metric_R2_3.markdown("<h1 style='text-align: left; vertical-align: bottom;color: Green; font-size: 150%; opacity: 0.7'> ✖️ Below market  </h1>" "  \n"  "The lower robutness, the lesser accurate the standard model make pay explaination and predictions. In general, we can improve robustness by including additional pay factors not captured by standard model, such as high potential, cost center, skills, etc. Please contact us for a free consultation.", unsafe_allow_html=True)
                
            # metric_R2_3.markdown("<h1 style='text-align: center; vertical-align: bottom; color: Black; background-color: #3498DB; opacity: 0.7; border-style: dotted'>Observation</h1>", unsafe_allow_html=True)
            
            # Show Net Gap
            metric_net_gap_1, metric_net_gap_2, metric_net_gap_3 = main_page.columns((1, 1, 1))            
            metric_net_gap_1.markdown("<h1 style='text-align: left; vertical-align: bottom; font-size: 200%; color: #3498DB; opacity: 0.7'>Gender Net Gap</h1>", unsafe_allow_html=True)
            metric_net_gap_1.plotly_chart(fig_net_gender_gap, use_container_width=True)
            
            metric_net_gap_2.markdown("<h1 style='text-align: left; vertical-align: bottom;color: #3498DB; font-size: 200%; opacity: 0.7'>Benchmark</h1>", unsafe_allow_html=True)
            metric_net_gap_2.markdown("<h1 style='text-align: left; vertical-align: bottom;color: Green; font-size: 150%; opacity: 0.7'> 🌐 <5% </h1>" "  \n"  "Model Robutness measures how well the pay factors drive pay decisions. For example 80% means pay factors explain 80 percent of the pay variation among employees.", unsafe_allow_html=True)
            
            metric_net_gap_3.markdown("<h1 style='text-align: left; vertical-align: bottom;color: #3498DB; font-size: 200%; opacity: 0.7'>Observation</h1>", unsafe_allow_html=True)
            if r2>=0.7:
                metric_net_gap_3.markdown("<h1 style='text-align: left; vertical-align: bottom;color: Green; font-size: 150%; opacity: 0.7'> ✔️ Align with market  </h1>" "  \n"  "The higher robustness, the more accurate the model make pay explination and predictions", unsafe_allow_html=True)
            else:
                metric_net_gap_3.markdown("<h1 style='text-align: left; vertical-align: bottom;color: Green; font-size: 150%; opacity: 0.7'> ✖️ Below market  </h1>" "  \n"  "The lower robutness, the lesser accurate the standard model make pay explaination and predictions. In general, we can improve robustness by including additional pay factors not captured by standard model, such as high potential, cost center, skills, etc. Please contact us for a free consultation.", unsafe_allow_html=True)
            
        if m_col2_but:
            main_page.empty()