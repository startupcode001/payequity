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

# Set functions
@st.experimental_memo(show_spinner=False)
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

# Function
@st.experimental_memo(show_spinner=False)
# Download Excel Template
def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    # href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{bin_file}">{file_label}</a>'
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">{file_label}</a>'
    return href

@st.experimental_memo(show_spinner=False)
# Run Goal Seek for insignificant gap and 0 gap
def reme_gap_seek(df,budget_df,X_full, project_group_feature, protect_group_class, seek_goal):
    factor_range = np.arange(2, -2,-0.005)
    threshold = 0.0005
    
    seek_budget_df = np.nan
    seek_budget = np.nan
    seek_resulting_gap  = np.nan
    seek_resulting_pvalues =  np.nan
    seek_adj_count = np.nan
    seek_adj_budget_pct = np.nan
    seek_success = False
        
    for factor in factor_range:
        budget_df, budget, resulting_gap, resulting_pvalues, adj_count, adj_budget_pct = reme(df,budget_df,X_full,factor, project_group_feature, protect_group_class)
        
        if np.abs(resulting_gap-seek_goal)<=threshold:
            seek_budget_df = budget_df
            seek_budget = budget
            seek_resulting_gap  = resulting_gap
            seek_resulting_pvalues =  resulting_pvalues
            seek_adj_count = adj_count
            seek_adj_budget_pct = adj_budget_pct
            seek_success = True
            
            print('Found factor that close gap:' + str(factor))
            print('Final Gap is '+str(seek_resulting_gap))
            print('Final p_value is '+str(seek_resulting_pvalues))
            print('Final Budget is '+str(seek_budget))
            print('Final Budget % '+str(seek_adj_budget_pct))
            break
        
    if seek_budget == np.nan:
        print('no result found')
        seek_success = False
    
    return seek_budget_df,seek_budget,seek_resulting_gap,seek_resulting_pvalues,seek_adj_count, seek_adj_budget_pct,seek_success
    
@st.experimental_memo(show_spinner=False)
# Run Goal Seek for insignificant gap and 0 gap
def reme_pvalue_seek(df,budget_df,X_full, project_group_feature, protect_group_class, seek_goal, current_pvalue):
    
    factor_range = np.arange(2, -2,-0.005)
    threshold = 0.0005
    
    seek_budget_df = np.nan
    seek_budget = np.nan
    seek_resulting_gap  = np.nan
    seek_resulting_pvalues =  np.nan
    seek_adj_count = np.nan
    seek_adj_budget_pct = np.nan
    seek_success = False
    
    if current_pvalue>=0.05:
        print('Current P value already greater than 5%: '+str(current_pvalue))
        seek_success = False
    else:
        for factor in factor_range:
            budget_df, budget, resulting_gap, resulting_pvalues, adj_count, adj_budget_pct = reme(df,budget_df,X_full,factor, project_group_feature, protect_group_class)

            if np.abs(resulting_pvalues-seek_goal)<=threshold:
                seek_budget_df = budget_df
                seek_budget = budget
                seek_resulting_gap  = resulting_gap
                seek_resulting_pvalues =  resulting_pvalues
                seek_adj_count = adj_count
                seek_adj_budget_pct = adj_budget_pct
                seek_success = True

                print('Found factor that close gap:' + str(factor))
                print('Final Gap is '+str(seek_resulting_gap))
                print('Final p_value is '+str(seek_resulting_pvalues))
                print('Final Budget is '+str(seek_budget))
                print('Final Budget % '+str(seek_adj_budget_pct))
                break

        if seek_budget == np.nan:
            print('no result found')
            seek_success = False
    
    return seek_budget_df,seek_budget,seek_resulting_gap,seek_resulting_pvalues,seek_adj_count, seek_adj_budget_pct,seek_success
    
# Set Style

# m = st.markdown("""
#     <style>
#     div.stButton > button:first-child {
#         background-color: #0099ff;
#         color:#ffffff;
#     }
#     div.stButton > button:hover {
#         background-color: #00ff00;
#         color:#ff0000;
#         }
#     </style>""", unsafe_allow_html=True)   
    
# Navigation Bar

st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

# Set Style
# m = st.markdown("""
#     <style>
#     div.stButton > button:first-child {box-shadow: 0px 0px 0px 2px #9fb4f2;background-color:#3498DB;border-radius:10px;border:1px solid #3498DB;display:inline-block;cursor:pointer;color:#ffffff;font-family:Arial;font-size:13px;padding:12px 37px;text-decoration:none;text-shadow:0px 1px 0px #283966;
#     &:hover {background-color:#476e9e;}
#     &:active {position:relative;top:1px;}}
#     </style>""", unsafe_allow_html=True)


# UI *********************************
# 'Hello :sunglasses: :heart: '

# st.dataframe(budget_df.head(4))
# st.dataframe(X_full)


# Setup Session State:
# if 'demo_run' not in st.session_state:
#     st.session_state['demo_run'] = 'no'

# Side Panel
st.sidebar.header('Start here')
st.sidebar.markdown(get_binary_file_downloader_html(demo_path, 'Download Input Template'), unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader('Upload your input Excel file', type=['xlsx'])
st.sidebar.markdown("""---""")

# Main Panel
c1, c2 = st.columns((1.5, 1))
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
            # st.session_state.demo_run = 'yes'
            with st.spinner('Running model, Please wait for it...'):
                # Demo Run
                m_info = main_page_info.success('Initialize Data')
                df_demo = pd.read_excel(demo_path,sheet_name="Submission")
                
                # Run discovery model: demo
                m_info = main_page_info.success('Running Gap Analysis')
                df, df_org, message, exclude_col, r2_raw, female_coff_raw, female_pvalue_raw, r2, female_coff, female_pvalue, before_clean_record, after_clean_record,hc_female,fig_r2_gender_gap,fig_raw_gender_gap,fig_net_gender_gap,X_full,budget_df = run(df_demo)
                
                # Run Reme Pvalue = 7%
                m_info = main_page_info.success('Running Remediation Statistical Significant')
                seek_budget_df,seek_budget,seek_resulting_gap,seek_resulting_pvalues,seek_adj_count, seek_adj_budget_pct,seek_success = reme_pvalue_seek(df,budget_df,X_full, project_group_feature='GENDER', protect_group_class='F', seek_goal=0.07, current_pvalue = female_coff)
                # Run Reme Zero Gap
                m_info = main_page_info.success('Running Remediation Zero Gap')
                seek_budget_df,seek_budget,seek_resulting_gap,seek_resulting_pvalues,seek_adj_count, seek_adj_budget_pct,seek_success = reme_gap_seek(df,budget_df,X_full, project_group_feature='GENDER', protect_group_class='F', seek_goal=0)
                
                # Run data validation
                m_info = main_page_info.success('Output Data Validation')
                demo_validation = convert_df(df_org)
                
            # Display run is successful message    
            m_info = main_page_info.success('View Demo: '+message.loc[['OVERVIEW']][0])
            
            main_page.markdown("""---""")
            m_col1_but_col1, m_col1_but_col2, m_col1_but_col3, m_col1_but_col4 = main_page.columns((2, 2, 2, 1))
            
            # Display headcount, Successful Run, Female Percent, download validation file
            m_col1_but_col1.metric('💬 Submission Record',before_clean_record)
            m_col1_but_col2.metric('🏆 Successful Run',after_clean_record)
            m_col1_but_col3.metric('👩 Female %',round(hc_female/after_clean_record,2)*100)
            m_col1_but_col4.download_button('📥 Download exclusions', data=demo_validation, file_name='Data Validation.csv')
            
            # Show R2
            metric_R2_1, metric_R2_2, metric_R2_3 = main_page.columns((1, 1.7, 1.7))            
            metric_R2_1.markdown("<h1 style='text-align: left; vertical-align: bottom; font-size: 200%; color: #3498DB; opacity: 0.7'>Robustness</h1>", unsafe_allow_html=True)
            metric_R2_1.pyplot(fig_r2_gender_gap, use_container_width=True)
            
            metric_R2_2.markdown("<h1 style='text-align: left; vertical-align: bottom;color: #3498DB; font-size: 200%; opacity: 0.7'>Benchmark</h1>", unsafe_allow_html=True)
            metric_R2_2.markdown("<h1 style='text-align: left; vertical-align: bottom;color: Green; font-size: 150%; opacity: 0.7'> 🌐 70% ~ 100%  </h1>" "  \n"  "Model Robutness measures how well the pay factors drive pay decisions. For example 80% means pay factors explain 80 percent of the pay variation among employees.", unsafe_allow_html=True)
            
            metric_R2_3.markdown("<h1 style='text-align: left; vertical-align: bottom;color: #3498DB; font-size: 200%; opacity: 0.7'>Observation</h1>", unsafe_allow_html=True)
            if r2>=0.7:
                metric_R2_3.markdown("<h1 style='text-align: left; vertical-align: bottom;color: Green; font-size: 150%; opacity: 0.7'> ✔️ Align with market  </h1>" "  \n"  "The higher robustness, the more accurate the model make pay explination and predictions", unsafe_allow_html=True)
            else:
                metric_R2_3.markdown("<h1 style='text-align: left; vertical-align: bottom;color: Orange; font-size: 150%; opacity: 0.7'> ⚠️ Below market  </h1>" "  \n"  "The lower robutness, the lesser accurate the standard model make pay explaination and predictions. In general, we can improve robustness by including additional pay factors not captured by standard model, such as high potential, cost center, skills, etc. Please contact us for a free consultation.", unsafe_allow_html=True)
                
            # metric_R2_3.markdown("<h1 style='text-align: center; vertical-align: bottom; color: Black; background-color: #3498DB; opacity: 0.7; border-style: dotted'>Observation</h1>", unsafe_allow_html=True)
            
            # Show Net Gap
            metric_net_gap_1, metric_net_gap_2, metric_net_gap_3 = main_page.columns((1, 1.6, 1.6))            
            metric_net_gap_1.markdown("<h1 style='text-align: left; vertical-align: bottom; font-size: 200%; color: #3498DB; opacity: 0.7'>Gender Net Gap</h1>", unsafe_allow_html=True)
            metric_net_gap_1.plotly_chart(fig_net_gender_gap, use_container_width=True)
            
            metric_net_gap_2.markdown("<h1 style='text-align: left; vertical-align: bottom;color: #3498DB; font-size: 200%; opacity: 0.7'>Benchmark</h1>", unsafe_allow_html=True)
            metric_net_gap_2.write("<h1 style='text-align: left; vertical-align: bottom;color: Green; font-size: 150%; opacity: 0.7'> 🌐 -5% ~ 0%</h1>" "For every 1 dollar paid to male employees, how much (lesser)/more is paid to female employees. For example -3% means on average female employees is paid 3% LOWER than male employees all else equal. Typically the net pay gap in US is under -5%", unsafe_allow_html=True)
            
            metric_net_gap_3.markdown("<h1 style='text-align: left; vertical-align: bottom;color: #3498DB; font-size: 200%; opacity: 0.7'>Observation</h1>", unsafe_allow_html=True)
            if female_coff>=-0.05:
                metric_net_gap_3.markdown("<h1 style='text-align: left; vertical-align: bottom;color: Green; font-size: 150%; opacity: 0.7'> ✔️ Align with market  </h1>" "  \n"  "The smaller the pay gap, the better pay equity between genders", unsafe_allow_html=True)
            else:
                metric_net_gap_3.markdown("<h1 style='text-align: left; vertical-align: bottom;color: Orange; font-size: 100%; opacity: 0.7'> ⚠️ Below market  </h1>" "The larger the pay gap (negative), the bigger pay inequity between genders", unsafe_allow_html=True)
                metric_net_gap_3.markdown("<h1 style='text-align: left; vertical-align: bottom; color: Orange; font-size: 100%; opacity: 0.7'> 📊 Yes - Statistical Significant  </h1>" "The larger the pay gap (negative), the bigger pay inequity between genders", unsafe_allow_html=True)
                
            main_page.markdown("""---""")
            overview_1, overview_2 = main_page.columns((1, 3))
            overview_1.image('Picture/overview.jpg',use_column_width='auto')
            overview_2.write('The lower robutness, the lesser accurate the standard model make pay explaination and predictions. In general, we can improve robustness by including additional pay factors not captured by standard model, such as high potential, cost center, skills, etc. Please contact us for a free consultation',use_column_width='auto')
            
            main_page.markdown("""---""")
            lm, s_Cur, s_A, s_B, rm = main_page.columns((1, 1, 1, 1))
            # scenario_A.button('📥 Download exclusions')
            
            # Run remediation model: demo goal seek on gap        
            s1.markdown(seek_budget)
            s1.success('Done!')

#             print('Discovery model p value is: '+str(female_pvalue))

#             seek_budget_df,seek_budget,seek_resulting_gap,seek_resulting_pvalues,seek_adj_count, seek_adj_budget_pct,seek_success = reme_pvalue_seek(df,budget_df,X_full, project_group_feature='GENDER', protect_group_class='F', seek_goal=0.07, current_pvalue = female_pvalue)

            
        if m_col2_but:
            main_page.empty()

