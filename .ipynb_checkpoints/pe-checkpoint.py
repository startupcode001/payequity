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
# locale.setlocale(locale.LC_ALL, 'en_US')

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

# UI *********************************
# 'Hello :sunglasses: :heart: '

# st.dataframe(budget_df.head(4))
# st.dataframe(X_full)


# Setup Session State:
# if 'demo_run' not in st.session_state:
#     st.session_state['demo_run'] = 'no'
# Side Panel
st.sidebar.header(' 🔔 Start here')
st.sidebar.markdown(get_binary_file_downloader_html(demo_path, 'Step 1: Download Instruction and Data Template'), unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader('Step 2: Upload Data Template', type=['xlsx'])
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
    
    if uploaded_file is not None:
        main_page_info.info('Use input file.')
    else:
        m_info = main_page_info.info('Awaiting the upload of the input file.')
        m_col1,m_col2,t1 = main_page.columns((1, 1, 2))
        
        m_col1_but = m_col1.button('See demo')
        m_col2_but = m_col2.button('Close demo')
        
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
                seek_budget_df_pv,seek_budget_pv,seek_resulting_gap_pv,seek_resulting_pvalues_pv,seek_adj_count_pv, seek_adj_budget_pct_pv,seek_pass_pv, seek_success_pv = reme_pvalue_seek(df,budget_df,X_full, project_group_feature='GENDER', protect_group_class='F', seek_goal=0.07, current_pvalue = female_coff)
                
                # Run Reme Zero Gap
                m_info = main_page_info.success('Running Remediation Zero Gap')
                seek_budget_df_gap,seek_budget_gap,seek_resulting_gap_gap,seek_resulting_pvalues_gap,seek_adj_count_gap, seek_adj_budget_pct_gap,seek_pass_gap, seek_success_gap = reme_gap_seek(df,budget_df,X_full, project_group_feature='GENDER', protect_group_class='F', seek_goal=0, current_gap = female_coff)
                
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
            metric_R2_1.markdown("<h1 style='text-align: left; vertical-align: bottom; font-size: 150%; color: #3498DB; opacity: 0.7'>Robustness</h1>", unsafe_allow_html=True)
            metric_R2_1.pyplot(fig_r2_gender_gap, use_container_width=True)
            
            metric_R2_2.markdown("<h1 style='text-align: left; vertical-align: bottom;color: #3498DB; font-size: 150%; opacity: 0.7'>Benchmark</h1>", unsafe_allow_html=True)
            metric_R2_2.markdown("<h1 style='text-align: left; vertical-align: bottom;color: Green; font-size: 110%; opacity: 0.7'> 🌐 70% ~ 100%  </h1>" "  \n"  "Model Robutness measures how well the pay factors drive pay decisions. For example 80% means pay factors explain 80 percent of the pay variation among employees.", unsafe_allow_html=True)
            
            metric_R2_3.markdown("<h1 style='text-align: left; vertical-align: bottom;color: #3498DB; font-size: 150%; opacity: 0.7'>Observation</h1>", unsafe_allow_html=True)
            if r2>=0.7:
                metric_R2_3.markdown("<h1 style='text-align: left; vertical-align: bottom;color: Green; font-size: 110%; opacity: 0.7'> ✔️ Align with market  </h1>" "  \n"  "The higher robustness, the more accurate the model make pay explination and predictions", unsafe_allow_html=True)
            else:
                metric_R2_3.markdown("<h1 style='text-align: left; vertical-align: bottom;color: Orange; font-size: 110%; opacity: 0.7'> ⚠️ Below market  </h1>" "  \n"  "The lower robutness, the lesser accurate the standard model make pay explaination and predictions. In general, we can improve robustness by including additional pay factors not captured by standard model, such as high potential, cost center, skills, etc. Please contact us for a free consultation.", unsafe_allow_html=True)
                
            # metric_R2_3.markdown("<h1 style='text-align: center; vertical-align: bottom; color: Black; background-color: #3498DB; opacity: 0.7; border-style: dotted'>Observation</h1>", unsafe_allow_html=True)
            
            # Show Net Gap
            metric_net_gap_1, metric_net_gap_2, metric_net_gap_3 = main_page.columns((1, 1.6, 1.6))            
            metric_net_gap_1.markdown("<h1 style='text-align: left; vertical-align: bottom; font-size: 150%; color: #3498DB; opacity: 0.7'>Gender Net Gap</h1>", unsafe_allow_html=True)
            metric_net_gap_1.plotly_chart(fig_net_gender_gap, use_container_width=True)
            
            metric_net_gap_2.markdown("<h1 style='text-align: left; vertical-align: bottom;color: #3498DB; font-size: 150%; opacity: 0.7'>Benchmark</h1>", unsafe_allow_html=True)
            metric_net_gap_2.write("<h1 style='text-align: left; vertical-align: bottom;color: Green; font-size: 110%; opacity: 0.7'> 🌐 > -5% </h1>" "For every 1 dollar paid to male employees, how much (lesser)/more is paid to female employees. For example -3% means on average female employees is paid 3% LOWER than male employees all else equal. Typically the net pay gap in US is between -5% and +1%", unsafe_allow_html=True)
            
            metric_net_gap_3.markdown("<h1 style='text-align: left; vertical-align: bottom;color: #3498DB; font-size: 150%; opacity: 0.7'>Observation</h1>", unsafe_allow_html=True)
            if female_coff>=-0.05:
                metric_net_gap_3.markdown("<h1 style='text-align: left; vertical-align: bottom;color: Green; font-size: 110%; opacity: 0.7'> ✔️ Align with market  </h1>" "  \n"  "The smaller the pay gap, the better pay equity between genders", unsafe_allow_html=True)
            else:
                metric_net_gap_3.markdown("<h1 style='text-align: left; vertical-align: bottom;color: Orange; font-size: 110%; opacity: 0.7'> ⚠️ Below market  </h1>" "The larger the pay gap (negative), the bigger pay inequity between genders", unsafe_allow_html=True)
                metric_net_gap_3.markdown("<h1 style='text-align: left; vertical-align: bottom; color: Orange; font-size: 110%; opacity: 0.7'> 📊 Yes - Statistical Significant  </h1>" "The larger the pay gap (negative), the bigger pay inequity between genders", unsafe_allow_html=True)
                
            main_page.markdown("""---""")
            overview_1, overview_2 = main_page.columns((1, 3))
            overview_1.image('Picture/overview.jpg',use_column_width='auto')
            overview_2.write('The lower robutness, the lesser accurate the standard model make pay explaination and predictions. In general, we can improve robustness by including additional pay factors not captured by standard model, such as high potential, cost center, skills, etc. Please contact us for a free consultation',use_column_width='auto')
            
            main_page.markdown("""---""")
            
            message_budget_pv = np.nan
            if seek_pass_pv == False:
                message_budget_pv = '0 - current gap is already statistically insignificant'
            elif (seek_pass_pv == True) and (seek_success_pv == False):
                message_budget_pv = 'No result is found, please contact consultant for more detail'
            else:
                message_budget_pv = str(locale.format("%d", round(seek_budget_pv/1000,0), grouping=True))+'K'
                
            message_budget_gap = np.nan
            if seek_pass_gap == False:
                message_budget_gap = '0 - current gap is greater than zero, no need for further adjustment'
            elif (seek_pass_gap == True) and (seek_success_gap == False):
                message_budget_gap = 'No result is found, please contact consultant for more detail'
            else:
                message_budget_gap = str(locale.format("%d", round(seek_budget_gap/1000,0), grouping=True))+'K'
            
            scenario = ['Current','A','B']
            action = ['No change','Reduce gender gap to statistically insignificant','Completely close gender gap']
            budget = ['0',message_budget_pv,message_budget_gap]
            net_gap = [female_coff,seek_resulting_gap_pv,seek_resulting_gap_gap]
            net_gap = [f'{i*100:.1f}%' for i in net_gap]
            
            # result_pvalue = [female_pvalue,seek_resulting_pvalues_pv,seek_resulting_pvalues_gap]
            
            df_reme = pd.DataFrame({'Scenario': scenario, 'Action': action, 'Remediation Budget': budget, 'Net Gender Gap': net_gap})

            cell_hover = {  # for row hover use <tr> instead of <td>
                            'selector': 'td:hover',
                            'props': [('background-color', 'lightgrey')]
                        }
            index_names = {
                            'selector': '.index_name',
                            'props': 'font-style: italic; color: darkgrey; font-weight:normal;'
                        }
            headers = {
                            'selector': 'th:not(.index_name)',
                            'props': 'background-color: #3498DB; color: white; text-align: center; '
                        }
            styler = df_reme.style.hide_index().set_table_styles([cell_hover, index_names, headers], overwrite=False)
            
            main_page.markdown("<h1 style='text-align: left; vertical-align: bottom;color: #3498DB; font-size: 150%; opacity: 0.7'>Remediation Scenarios</h1>", unsafe_allow_html=True)
            main_page.write(styler.to_html(), unsafe_allow_html=True)
            
#             reme_1, overview_2 = main_page.columns((1, 3))
            
#             lm, SCur, SA, SB, rm = main_page.columns((1, 1, 1, 1, 1))  
#             SCur.markdown('📓 Current')
#             SA.markdown('📘 Scenario A')
#             SB.markdown('📗 Scenario B')
            
#             lm, SCur_Act, SA_Act, SB_Act, rm = main_page.columns((1, 1, 1, 1, 1))  
#             lm.markdown("📌 Action", unsafe_allow_html=True)
#             SCur_Act.markdown('No Action')
#             SA_Act.markdown('Reduce gap to statistically insignificant')
#             SB_Act.markdown('Close gap to zero')
            
#             lm, SCur_Budget, SA_Budget, SB_Budget, rm = main_page.columns((1, 1, 1, 1, 1))  
#             message_budget_pv = np.nan
#             if seek_success_pv == False:
#                 message_budget_pv = 'Already statistically insignificant - No additional budget is required'
#             else:
#                 message_budget_pv = str(seek_budget_pv)
                
#             lm.markdown("💰 Budget", unsafe_allow_html=True)
#             SCur_Act.markdown('0')
#             SA_Act.markdown(message_budget_pv)
#             SB_Act.markdown(str(seek_budget_gap))
            
#             print('Discovery model p value is: '+str(female_pvalue))

#             seek_budget_df,seek_budget,seek_resulting_gap,seek_resulting_pvalues,seek_adj_count, seek_adj_budget_pct,seek_success = reme_pvalue_seek(df,budget_df,X_full, project_group_feature='GENDER', protect_group_class='F', seek_goal=0.07, current_pvalue = female_pvalue)

            
        if m_col2_but:
            main_page.empty()

