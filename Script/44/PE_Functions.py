import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
import operator
import copy 

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.graph_objects as go
import plotly.express as px
from streamlit_echarts import st_echarts
import hydralit_components as hc
from pathlib import Path

from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer,KNNImputer

from patsy import dmatrices
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from statsmodels.sandbox.regression.predstd import wls_prediction_std

from math import pi
import base64
import os

import xlsxwriter
from io import BytesIO

import locale

import pyrebase

import plost

from PE_Parameter import *

# Helper Functions Starts here #

# def rename_column(df):
#     df.columns = [c.strip().upper().replace(' ', '_') for c in df.columns]
#     df.columns = [c.strip().upper().replace('/', '_') for c in df.columns]
#     df.columns = [c.strip().upper().replace('-', '_') for c in df.columns]
#     df.columns = [c.strip().upper().replace('(', '') for c in df.columns]
#     df.columns = [c.strip().upper().replace(')', '') for c in df.columns]
#     df.columns = [c.strip().upper().replace('.', '') for c in df.columns]
#     df.columns = [c.strip().upper().replace('___', '_') for c in df.columns]
#     df.columns = [c.strip().upper().replace('__', '_') for c in df.columns]
#     return df
# st.set_page_config(layout="wide")

# Set Style
# style_path = Path(__file__).parents[0].__str__()+'/Style/style.css'
# with open(style_path) as f:
#     st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def rename_column(df):
    df.columns = [c.strip().replace(' ', '_') for c in df.columns]
    df.columns = [c.strip().replace('/', '_') for c in df.columns]
    df.columns = [c.strip().replace('-', '_') for c in df.columns]
    df.columns = [c.strip().replace('(', '') for c in df.columns]
    df.columns = [c.strip().replace(')', '') for c in df.columns]
    df.columns = [c.strip().replace('.', '') for c in df.columns]
    df.columns = [c.strip().replace('___', '_') for c in df.columns]
    df.columns = [c.strip().replace('__', '_') for c in df.columns]
    return df

def gender_name_replace(text):
    if ('F' in text):
        return 'Female'
    elif ('M' in text):
        return 'Male'
    elif ('N' in text):
        return 'Non-Binary'
    elif ('U' in text):
        return 'Unknown'
    else:
        return text

def plot_eth_donut(data):
    # explosion    
    # plt.clf()
    n_exp = len(data['ETHNICITY_NAME'])
    explode = np.repeat(0.02, n_exp).tolist()
    explode = tuple(explode)
    
    centre_circle = plt.Circle((0, 0), 0.65, fc='white')
    # fig_eth = plt.subplots()
    # fig_eth = plt.gcf()
    fig_eth = plt.figure(num = 0,dpi=100)
    # fig_eth, ax_eth = plt.subplots()
    
    plt.pie(data['HC'],  labels=data['ETHNICITY_NAME'],
            autopct='%1.1f%%', pctdistance=0.85,
            explode=explode)
    fig_eth.gca().add_artist(centre_circle)
    plt.tight_layout()
    # plt.title('Ethnicity by Headcount')
    # plt.figure(figsize=(500,500))
    return fig_eth

def plot_gender_donut(data):
    # explosion    
    # plt.clf()
    n_exp = len(data['GENDER_NAME'])
    explode = np.repeat(0.02, n_exp).tolist()
    explode = tuple(explode)
    # data.to_excel('gender_check.xlsx')
    
    # Pie Chart
    plt.pie(data['HC'],  labels=data['GENDER_NAME'],
            autopct='%1.1f%%', pctdistance=0.85,
            explode=explode)
    centre_circle = plt.Circle((0, 0), 0.65, fc='white')
    # fig_gender = plt.gcf()
    fig_gender = plt.figure(num = 1,dpi=100)
    # fig_gender, ax_gender = plt.subplots()
    fig_gender.gca().add_artist(centre_circle)
    plt.tight_layout()
    # plt.title('Ethnicity by Headcount')
    # plt.figure(figsize=(500,500))
    # plt.show()
    return fig_gender

def plot_bar(data,col):    
    scale_max = round(data['COEF_DISPLAY'].max(),2)+0.02
    scale_min = round(data['COEF_DISPLAY'].min(),2)-0.02
    fig = px.bar(data, x=col, y="COEF_DISPLAY", color='STAT', template='seaborn', text=[f'{round(100*j,1)}%' for i,j in zip(data[col],data['COEF_DISPLAY'])])
        
    fig.update_layout(showlegend=False,autosize=False, margin=dict(l=10, r=10, t=10, b=10),yaxis_range=[scale_min,scale_max],paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig.update_yaxes(automargin=True, showticklabels=False, visible=False)
    fig.update_xaxes(automargin=True, showticklabels=True, visible=True)
    # fig.update_xaxes(automargin=True, showticklabels=True, visible=True)
    # fig['layout']['yaxis'].update(autorange = True)s
    # fig['layout']['xaxis'].update(autorange = True)
    fig.update_traces(textfont_size=10, textangle=0, textposition="outside", cliponaxis=False)
    return fig

def clean_req_feature(data, feature, valid_feature_list, warning_message, data_type="string", 
                      val_col = 'VALIDATION_MESSAGE',val_flag_col = 'VALIDATION_FLAG'):
    da = data.copy()
    # Check exclusion and remove invalid data
    if (data_type == "numeric") and (feature == 'SALARY'):
        da[feature] = pd.to_numeric(da[feature], errors='coerce')
        da.loc[(da[feature].isna()) | (da[feature]<=0), val_flag_col] = 1
        exclude_feature_num = sum(da[feature].isna()*1)+sum((da[feature]<=0)*1)
    elif data_type == "numeric":
        # Numeric features, exclude nan, strings
        da[feature] = pd.to_numeric(da[feature], errors='coerce')
        exclude_feature_num = sum(da[feature].isna()*1)
        da.loc[da[feature].isna(), val_flag_col] = 1
    elif data_type == 'datetime':
        da[feature] = pd.to_datetime(da[feature], errors='coerce')
        exclude_feature_num = sum(da[feature].isna()*1)
        da.loc[da[feature].isna(), val_flag_col] = 1
#         da = da.dropna(subset=[feature])
    else:
        # String features, check for nan and valid list     
        if len(valid_feature_list) == 0: 
            # exclude nan only
            check_na_feature = sum(da[feature].isna()*1)/len(da[feature])
            exclude_feature_num = sum(da[feature].isna()*1)
            da.loc[da[feature].isna(), val_flag_col] = 1
#             da = da.dropna(subset=[feature])
        else:
            # exclude nan, and those not in valid_feature_list
            da[feature] = da[feature].str[0].str.upper()
            exclude_feature_num = sum(~(da[feature].isin(valid_feature_list))*1)
            da.loc[~da[feature].isin(valid_feature_list), val_flag_col] = 1
#             da = da[da[feature].isin(valid_feature_list)]

    # Record message
    if exclude_feature_num>0:
        warning_feature = feature+": exclude "+str(exclude_feature_num)+" invalid or blank records"
        da.loc[da[val_flag_col]==1, val_col] = da[val_col]+" | "+ warning_feature
    else:
        warning_feature = feature+": Successful Run"
    warning_message[feature] = warning_feature
#     print(warning_message)
    da[val_flag_col]=0
    return da,warning_message

def clean_optional_feature(data, feature, valid_feature_list, warning_message, exclude_col, data_type="string", 
                      val_col = 'VALIDATION_MESSAGE',val_flag_col = 'VALIDATION_FLAG'):
    da = data.copy()
    check_na_feature = sum(da[feature].isna()*1)/len(da[feature])
    
    # Step 1, determine if there is no input, if yes, then this is an optional factor and we should exclude from our model
    if check_na_feature == 1:
        # Step 1, determine if there is no input, if yes, then this is an optional factor and we should exclude from our model
        exclude_col.append(feature)
        warning_feature = feature + ": exclude this optional driver"
        warning_message[feature] = warning_feature
    else:
        # Step 2, clean up data just like the required features
        da, warning_message = clean_req_feature(data = da,feature = feature,valid_feature_list=valid_feature_list,
                                                warning_message = warning_message, data_type=data_type,val_col = val_col,val_flag_col = val_flag_col)
        
    return da, warning_message, exclude_col

def set_dropped_category(series, dropped_category):
    '''
    Function to move one category to the top of a category column, such that
    it is the one that gets dropped for the OLS
    '''

    temp = list(series.cat.categories)
    temp.insert(0, temp.pop(temp.index(dropped_category)))
    series = series.cat.reorder_categories(temp)

    return series

# Main Program Starts here #

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

# Download Excel Files
def get_excel_file_downloader_html(data, file_label='File'):
    bin_str = base64.b64encode(data).decode()
    # bin_str = base64.b64encode(data)
    # href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{bin_file}">{file_label}</a>'    
    # href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{file_label}">{file_label}</a>'
    href = f'<a href="data:file/xlsx;base64,{bin_str}" download="{file_label}">{file_label}</a>'
    # href = f'<a href="data:file/xlsx;base64,{b64}" download="new_file.{extension}">Download {extension}</a>'
    return href

def reme(df,budget_df,X_full,factor, project_group_feature, protect_group_class):
    budget_df['adj_lower'] = budget_df['predicted'] - factor * budget_df['pred_stderr']

    # Adjust protect group pay only, others set to original pay
    budget_df['adj_salary'] = budget_df['original']
    budget_df.loc[(budget_df[project_group_feature] == protect_group_class) & (budget_df['original'] < budget_df['adj_lower']),'adj_salary'] = budget_df['adj_lower']
    
    # Recalculate pay gap and p value with adjusted salary
    model = sm.OLS(budget_df['adj_salary'], X_full)
    results = model.fit()
    
#     X_full.to_excel('xfull.xlsx')
    
#     asdf
    
    budget = np.sum(np.exp(budget_df['adj_salary']) - np.exp(budget_df['original']))
    budget_df['S_Salary'] = np.trunc(np.exp(budget_df['original']))
    budget_df['S_Budget'] = np.trunc(np.exp(budget_df['adj_salary']))-np.trunc(np.exp(budget_df['original']))
    budget_df['S_Adjusted'] = np.trunc(np.exp(budget_df['adj_salary']))
    budget_df['S_AdjInd'] = 0
    budget_df.loc[budget_df['S_Budget'] >0, 'S_AdjInd']=1

    # Reporting
    current_total_salary = np.sum(budget_df['S_Salary'])
    Budget_PCT = budget_df['S_Budget']/np.exp(budget_df['original'])
    
    # target_position = 1
    # resulting_gap = results.params[target_position]
    # resulting_pvalues = results.pvalues[target_position]
    
    df_result = process_run_result(results)
    resulting_gap = df_result[df_result['CONTENT'] == protect_group_class]['COEF'].values[0]
    resulting_pvalues = df_result[df_result['CONTENT'] == protect_group_class]['PVALUE'].values[0]
    
    adj_count = budget_df['S_AdjInd'].sum()
    adj_average = Budget_PCT[Budget_PCT>0].mean()
    adj_max = Budget_PCT[Budget_PCT>0].max()
    adj_budget_pct = budget/current_total_salary
    
    
    # df_result.to_excel('run_result.xlsx')
    
    # print(resulting_gap)
    # print(resulting_pvalues)    
    
    # print(results.summary())
    # print("***")
    # print("Factor: "+str(factor))
    # print(resulting_gap)
    # print(resulting_pvalues)
    # print(adj_count)
    # print(adj_budget_pct)
    # budget_df.to_excel('check_final_budget.xlsx')
    # asdf

    return budget_df, budget, resulting_gap, resulting_pvalues, adj_count, adj_budget_pct, results

def process_run_result(results):
    df_result = results.summary2().tables[1]
    df_result.reset_index(level=0, inplace=True)
    df_result = df_result.rename(columns={"index":"CONTENT"})
    
    df_result = df_result[(df_result['CONTENT'].str.contains("GENDER")==True) | (df_result['CONTENT'].str.contains("ETHNICITY")==True)]
    
    df_result = df_result[['CONTENT','Coef.','P>|t|']]
    df_result.columns = ['CONTENT','COEF','PVALUE']
    # df_result_gender['GENDER'] = df_result_gender['CONTENT']
    df_result['CONTENT'] = df_result['CONTENT'].str.replace("[","")
    df_result['CONTENT'] = df_result['CONTENT'].str.replace("]","")
    
    df_result['FEATURE'] = 'GENDER'
    df_result.loc[(df_result['CONTENT'].str.contains("ETHNICITY")==True),'FEATURE'] = 'ETHNICITY'
    df_result['CONTENT'] = df_result['CONTENT'].str.split('.').str[-1]
    df_result['CONTENT_DISPLAY'] = df_result.apply(lambda x: gender_name_replace(text=x['CONTENT']),axis=1)
    df_result = df_result[~((df_result['CONTENT_DISPLAY']=='Unknown') | (df_result['CONTENT_DISPLAY']=='unknown'))]
    df_result["STAT_COUNT"]=0
    df_result.loc[((df_result['PVALUE']<=0.05) & (df_result['COEF']<0)) ,"STAT_COUNT"] = 1
    
    # df_result.to_excel('run_result.xlsx')
    # asdf
    
    return df_result
    # df_result = df_result.drop(columns=['CONTENT'])
    # df_result['COEF_DISPLAY'] = df_result['COEF']
    # df_result.to_excel('run_result.xlsx')

@st.experimental_memo(show_spinner=False)
# Run Goal Seek for insignificant gap and 0 gap
def reme_gap_seek(df,budget_df,X_full, project_group_feature, protect_group_class, seek_goal, current_pvalue, current_gap, search_step = -0.001):
    factor_range = np.arange(2, -2,search_step)
    threshold = 0.0005
    
    # seek_budget_df = np.nan
    seek_budget_df = pd.DataFrame()
    seek_budget = np.nan
    seek_resulting_gap  = np.nan
    seek_resulting_pvalues =  np.nan
    seek_adj_count = np.nan
    seek_adj_budget_pct = np.nan
    seek_pass = False
    seek_success = False
    
    if current_gap>=0:
        print('current gap is already >= 0')
        seek_pass = False
        seek_success = False
        seek_resulting_gap = current_gap
    else:
        seek_pass = True
        for factor in factor_range:
            budget_df, budget, resulting_gap, resulting_pvalues, adj_count, adj_budget_pct, results = reme(df,budget_df,X_full,factor, project_group_feature, protect_group_class)

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
                
                keep_list = ['EEID','S_Budget','S_Adjusted']
                seek_budget_df = seek_budget_df[keep_list]
                seek_budget_df.columns=['EEID','SCENARIO_B_ADJUSTMENT','SCENARIO_B_ADJUSTED_SALARY']
                # seek_budget_df = seek_budget_df.merge(df,on='EEID',how='inner')
                seek_budget_df.to_excel('budget_gap.xlsx')
                df_result = process_run_result(results)
                df_result.to_excel('zero_gap_result.xlsx')
                break

        if seek_budget == np.nan:
            print('no result found')
            seek_success = False
    
    return seek_budget_df,seek_budget,seek_resulting_gap,seek_resulting_pvalues,seek_adj_count, seek_adj_budget_pct,seek_pass,seek_success
    
@st.experimental_memo(show_spinner=False)
# Run Goal Seek for insignificant gap and 0 gap
def reme_pvalue_seek(df,budget_df,X_full, project_group_feature, protect_group_class, seek_goal, current_pvalue, current_gap, input_df_result, count_loop, search_step= -0.005):
    
    factor_range = np.arange(2, -2,search_step)
    threshold = 0.0005
    
    # seek_budget_df = np.nan
    seek_budget_df = pd.DataFrame()
    seek_budget = np.nan
    seek_resulting_gap  = np.nan
    seek_resulting_pvalues =  np.nan
    seek_adj_count = np.nan
    seek_adj_budget_pct = np.nan
    seek_pass = False
    seek_success = False
    
    if (current_pvalue>=0.05) | (current_gap>0):
        print('Current P value already greater than 5%: '+str(current_pvalue))
        print('Or Current gap already greater than 0%: '+str(current_gap))
        seek_pass = False
        seek_success = False
        seek_budget_df = copy.deepcopy(budget_df)
        seek_budget = 0
        seek_resulting_gap = current_gap
        seek_resulting_pvalues = current_pvalue
        seek_adj_count = 0
        seek_adj_budget_pct = 0
        df_result = copy.deepcopy(input_df_result)
    else:
        seek_pass = True
        for factor in factor_range:
            result_budget_df, result_budget, resulting_gap, resulting_pvalues, adj_count, adj_budget_pct, results = reme(df,budget_df,X_full,factor, project_group_feature, protect_group_class)

            # if np.abs(resulting_pvalues-seek_goal)<=threshold:
            if resulting_pvalues>=seek_goal:
                seek_budget_df = copy.deepcopy(result_budget_df)
                seek_budget = result_budget
                seek_resulting_gap  = resulting_gap
                seek_resulting_pvalues =  resulting_pvalues
                seek_adj_count = adj_count
                seek_adj_budget_pct = adj_budget_pct
                seek_success = True

                print('Found factor that close pvalue:' + str(factor))
                print('Final Gap is '+str(seek_resulting_gap))
                print('Final p_value is '+str(seek_resulting_pvalues))
                print('Final Budget is '+str(seek_budget))
                print('Final Budget % '+str(seek_adj_budget_pct))

                # keep_list = ['EEID','S_Budget','S_Adjusted']
                # seek_budget_df = seek_budget_df[keep_list]
                scenario_ind_name = 'SCENARIO_B_IND_'+str(count_loop)+"_"+protect_group_class
                scenario_increase_name = 'SCENARIO_B_BUDGET_'+str(count_loop)+"_"+protect_group_class
                scenario_adjusted_name = 'SCENARIO_B_ADJUSTED_SALARY_'+str(count_loop)+"_"+protect_group_class
                scenario_adjusted_log_name = 'LOG_SCENARIO_B_ADJUSTED_SALARY_'+str(count_loop)+"_"+protect_group_class
                
                seek_budget_df[scenario_ind_name]=seek_budget_df['S_AdjInd']
                seek_budget_df[scenario_increase_name]=seek_budget_df['S_Budget']
                seek_budget_df[scenario_adjusted_name]=seek_budget_df['S_Adjusted']
                seek_budget_df[scenario_adjusted_log_name]=seek_budget_df['adj_salary']
                # seek_budget_df.columns=['EEID','SCENARIO_A_ADJUSTMENT','SCENARIO_A_ADJUSTED_SALARY']
                
                seek_budget_df['original'] = seek_budget_df['adj_salary']
                # seek_budget_df = seek_budget_df.merge(df,on='EEID',how='inner')
                seek_budget_df.to_excel('budget_pv.xlsx')
                df_result = process_run_result(results)
                df_result.to_excel('pvalue_result.xlsx')
                break

        if seek_budget == np.nan:
            print('no result found')
            seek_success = False
    
    return seek_budget_df,seek_budget,seek_resulting_gap,seek_resulting_pvalues,seek_adj_count, seek_adj_budget_pct,seek_pass,seek_success,df_result

# def exam_col(file_path,display_path):
#     message = ""
#     df['Data2'].replace(r'^\s*$', np.nan, regex=True).isna().all()

def run(data=None, df_gender_name=None, req_list=None, ci = 0.95):
    # 1.1 Setup ************************************
    company_name = 'Client Name'
    version = 'Base Pay'
    starting_salary_flag = 'N'
    target = 'SALARY'

    # Setup error message to display on UI
    error_message = {}
    warning_message = {}
    exclude_col = []
    
    # Read Data
    try:
        df = data
    except:
        error_file_read = "Unable to read submission file, Please download and update data template again"
        error_message['File_Read'] = error_file_read

    df = rename_column(df)
    
    df_type = df.iloc[0]
    df = df.iloc[1:]
    
    # 2.1 Data Cleaning ****************************
    # Data Validation
    df['VALIDATION_MESSAGE']=""
    df['VALIDATION_FLAG']=0
    # Snapshot Date
    snapshot = np.nan
    try:
        df['SNAPSHOT_DATE'] = df['SNAPSHOT_DATE'].astype("string")
        snapshot = df['SNAPSHOT_DATE'].mode().tolist()[0]
    except:
        print('date exception')
        snapshot = date.today()
        error_snapshot = "Invalid snapshot date, Please check submission format is mm/dd/yyyy in data template"
        error_message['SNAPSHOT_DATE'] = error_snapshot
    df['NOW'] = pd.to_datetime(snapshot)
        
    # 2.2 Clean up All features ******************
    df,warning_message = clean_req_feature(data = df,feature = "EEID",valid_feature_list=[],warning_message = warning_message, data_type="string")
    df,warning_message = clean_req_feature(data = df,feature = "GENDER",valid_feature_list=["F","M","N","U","O"],warning_message = warning_message, data_type="string")
    df,warning_message = clean_req_feature(data = df,feature = "SALARY",valid_feature_list=[],warning_message = warning_message, data_type='numeric')
    df,warning_message = clean_req_feature(data = df,feature = "JOB_LEVEL_OR_COMP_GRADE",valid_feature_list=[],warning_message = warning_message, data_type="string")
    df,warning_message = clean_req_feature(data = df,feature = "JOB_FUNCTION",valid_feature_list=[],warning_message = warning_message,data_type="string")
    df,warning_message = clean_req_feature(data = df,feature = "COUNTRY",valid_feature_list=[],warning_message = warning_message,data_type="string")
    df,warning_message = clean_req_feature(data = df,feature = "LOCATION",valid_feature_list=[],warning_message = warning_message,data_type="string")
    df,warning_message = clean_req_feature(data = df,feature = "FULL_TIME",valid_feature_list=["Y","N"],warning_message = warning_message,data_type="string")

    # Clean up customized features
    standard_col = req_list
    all_col = df.columns.tolist()
    cust_col = [x for x in all_col if x not in standard_col]
    df_type = pd.DataFrame(df_type).reset_index()
    df_type.columns = ['COL_NAME','TYPE']
    df_type = df_type[~df_type['COL_NAME'].isin(standard_col)]
    for i, row in df_type.iterrows():
        df,warning_message, exclude_col = clean_optional_feature(data = df,feature = row['COL_NAME'],valid_feature_list=[],warning_message = warning_message,exclude_col = exclude_col, data_type=row['TYPE'])
    
    # Record Message
    df_org = df.copy()
    # df_org.to_excel('Data\data_validate.xlsx')
    before_clean_record = df_org.shape[0]
    
    df = df_org[df_org['VALIDATION_MESSAGE']==""]
    after_clean_record=df.shape[0]
    
    df_validation = df_org[df_org['VALIDATION_MESSAGE']!=""]
    list_validation = [x for x in df_validation.columns.tolist() if x not in ['VALIDATION_MESSAGE','VALIDATION_FLAG','NOW']]
    list_validation = ['VALIDATION_MESSAGE']+list_validation
    df_validation = df_validation[list_validation]
    
    warning_message['OVERVIEW'] = "OVERVIEW:Successfully run "+str(after_clean_record)+" out of "+str(before_clean_record)+" records"
    message = pd.DataFrame.from_dict(warning_message,orient='index')
    message.columns=['NAME']
    message = message['NAME'].str.split(":",expand = True)
    message.columns=['NAME','Message']
    message = message['Message']
    
    # 2.3 Feature Engineering ****************************
    # df_saveID = df['EEID']
    # df = df.set_index('EEID', drop=True)
    # col_list=df.columns.tolist()
    filter_list = [x for x in df.columns.tolist() if x not in exclude_col]
    df = df[filter_list]
    col_list=df.columns.tolist()
    
    print(exclude_col)
    print(filter_list)
    extra = ['VALIDATION_MESSAGE', 'VALIDATION_FLAG', 'NOW']
    exclude_feature = exclude_col    
    include_feature = [x for x in filter_list if x not in extra]
    
    # Calculate Age and Tenure
    try:
        df['AGE'] = np.ceil((df['NOW'] - df['DATE_OF_BIRTH']).dt.days/365)
    except:
        print('AGE: Calculation error')
    try:
        df['TENURE'] = np.ceil((df['NOW'] - df['DATE_OF_HIRE']).dt.days/365)
    except:
        print('TENURE: Calculation error')
    
    numeric_columns = ['SALARY','AGE', 'TENURE']
    if 'AGE' not in col_list:
        numeric_columns.remove('AGE')
    if 'TENURE' not in col_list:
        numeric_columns.remove('TENURE')
    
    # Include customized columns
    numeric_columns_customized = []
    for i, row in df_type.iterrows():
        if row['TYPE'] == 'numeric':
            numeric_columns_customized.append(row['COL_NAME'])
    
    numeric_columns_customized_include = [x for x in numeric_columns_customized if x in include_feature]
    numeric_columns.extend(numeric_columns_customized_include)
    numeric_columns = list(set(numeric_columns))
    
    for c in numeric_columns:
        df.loc[df[c]=='na', c] = np.nan
        df[c] = pd.to_numeric(df[c])
    
    # df.to_excel('edu.xlsx')
    
    # %% Convert string columns to categories
    category_columns = [x for x in col_list if x not in numeric_columns]
    print(category_columns)
    for c in category_columns:
        df[c] = df[c].astype(str).str.strip().astype('category')
    
    feature = []
    baseline = []
    
    df['LOG_SALARY'] = np.log(df['SALARY'])

    # Default to category level w/ highest n-count
    cat_cols = list(df.select_dtypes(['category']).columns)
    for x in cat_cols:
        drop_cate = df[x].value_counts().index.tolist()[0]
        #Set Manually
        if x=='GENDER':
            print('Override! Gender',"level dropped:",'M')
            drop_cate = 'M'
        df[x] = set_dropped_category(df[x], drop_cate)
        feature.append(x)
        baseline.append(drop_cate)

    df_baseline = pd.DataFrame({'Features': feature,'Baseline': baseline})
    print('\n')
    print(df_baseline.to_markdown())
    
    # 3 Gender Modeling - Linear Regression ****************************
    col_list = df.columns
    
    # Remove all excess columns
    add_exclude_col = ['EEID','SALARY','SNAPSHOT_DATE', 'VALIDATION_MESSAGE', 'VALIDATION_FLAG', 'NOW','DATE_OF_BIRTH','DATE_OF_HIRE']
    add_exclude_col_predict = add_exclude_col+['GENDER','ETHNICITY']
    exclude_col = exclude_col+add_exclude_col
    exclude_col_predict = exclude_col+add_exclude_col_predict

    model_col = [x for x in col_list if x not in exclude_col]
    model_col_predict = [x for x in col_list if x not in exclude_col_predict]
    
    # Factors
    f_raw = 'LOG_SALARY ~ GENDER'
    f_discover = model_col[-1] + ' ~ ' + ' + '.join(map(str, model_col[0:len(model_col)-1]))
    f_predict= model_col[-1] + ' ~ ' + ' + '.join(map(str, model_col_predict[0:len(model_col_predict)-1]))
    
    print(f_raw)
    print(f_discover)
    print(f_predict)
    
    # Gender Raw Gap
    y_raw, x_raw = dmatrices(f_raw, df, return_type='dataframe')
    model_raw = sm.OLS(y_raw, x_raw)
    results = model_raw.fit()

    df_result = results.summary2().tables[1]
    df_result.reset_index(level=0, inplace=True)
    df_result = df_result.rename(columns={"index":"CONTENT"})

    r2_raw = results.rsquared
    r2_adj = results.rsquared_adj

    female_coff_raw = df_result.loc[df_result['CONTENT']=="GENDER[T.F]"]['Coef.'].tolist()[0]
    female_pvalue_raw = df_result.loc[df_result['CONTENT']=="GENDER[T.F]"]['P>|t|'].tolist()[0]
    
    # Gender Net Gap
    y_dis, x_dis = dmatrices(f_discover, df, return_type='dataframe')
    model_dis = sm.OLS(y_dis, x_dis)
    results = model_dis.fit()
    
    y_dis.to_excel('y_dis.xlsx')
    x_dis.to_excel('x_dis.xlsx')

    df_result = results.summary2().tables[1]
    df_result.reset_index(level=0, inplace=True)
    df_result = df_result.rename(columns={"index":"CONTENT"})
    
    df_result.to_excel('dis_model.xlsx')
    
    asdf
    
    r2 = results.rsquared
    r2_adj = results.rsquared_adj

    female_coff = df_result.loc[df_result['CONTENT']=="GENDER[T.F]"]['Coef.'].tolist()[0]
    female_pvalue = df_result.loc[df_result['CONTENT']=="GENDER[T.F]"]['P>|t|'].tolist()[0]
    
    nonb_coff = df_result.loc[df_result['CONTENT']=="GENDER[T.N]"]['Coef.'].tolist()[0]
    nonb_pvalue = df_result.loc[df_result['CONTENT']=="GENDER[T.N]"]['P>|t|'].tolist()[0]
    
    # Gender Net Gap - Prediciton
    y_predict, x_predict = dmatrices(f_predict, df, return_type='dataframe')
    model_predict = sm.OLS(y_predict, x_predict)
    results_predict = model_predict.fit()

    y_pred = results_predict.predict(x_predict)
    std, lower, upper = wls_prediction_std(results_predict,alpha=0.05)
    std_view,lower_view, upper_view = wls_prediction_std(results_predict,alpha=1-ci)
    ci_view_lower = "Lower_CI"+str(int(ci*100))
    ci_view_upper = "Upper_CI"+str(int(ci*100))
    
    print(ci_view_upper)
    
#     std_90, lower_90, upper_90 = wls_prediction_std(results_predict,alpha=0.10)
#     std_85, lower_85, upper_85 = wls_prediction_std(results_predict,alpha=0.15)
#     std_80, lower_80, upper_80 = wls_prediction_std(results_predict,alpha=0.20)
#     std_75, lower_75, upper_75 = wls_prediction_std(results_predict,alpha=0.25)
#     std_70, lower_70, upper_70 = wls_prediction_std(results_predict,alpha=0.30)
    
    # Save budget file for prediction
    X_full = x_dis    
    
    if 'ETHNICITY' in include_feature:
        budget_df = pd.DataFrame({'EEID':df['EEID'], 'original_before': df['LOG_SALARY'], 'original': df['LOG_SALARY'], 'GENDER': df['GENDER'],'ETHNICITY': df['ETHNICITY'], 'predicted':y_pred,'pred_lower': lower, 'pred_upper': upper, 'pred_stderr': std})
    else:
        budget_df = pd.DataFrame({'EEID':df['EEID'], 'original_before': df['LOG_SALARY'], 'original': df['LOG_SALARY'], 'GENDER': df['GENDER'],'predicted':y_pred,'pred_lower': lower, 'pred_upper': upper, 'pred_stderr': std})
    predict_df = pd.DataFrame({'EEID':df['EEID'] ,'Predicted Salary':np.trunc(np.exp(y_pred)), 'Lower_CI95': np.trunc(np.exp(lower)), 'Upper_CI95': np.trunc(np.exp(upper)), ci_view_lower: np.trunc(np.exp(lower_view)), ci_view_upper: np.trunc(np.exp(upper_view))})
    
#     X_full.to_excel('xfull.xlsx')
#     df.to_excel('check_final.xlsx')
    budget_df.to_excel('check_final_budget.xlsx')
    predict_df.to_excel('check_final_predict.xlsx')
    
    # r2 = 0.91
    # Graphs
    # fig_r2_gender_gap = plot_full_pie(r2,'r2')
    # fig_r2_gender_gap = plot_r2(r2)
    # fig_raw_gender_gap = plot_gender_gap(female_coff)
    # fig_net_gender_gap = plot_gender_gap(female_coff)
    
    # Statistics for output
    hc_female = df[(df['GENDER']=='F') | (df['GENDER']=='FEMALE')].shape[0]
    avg_pay = df['SALARY'].mean()
    
    # Result Table
    df_result.to_excel('result_regression.xlsx')
    
    # Result Gender Table
    df_result_gender = df_result[df_result['CONTENT'].str.contains("GENDER")==True]
    df_result_gender = df_result_gender[['CONTENT','Coef.','P>|t|']]
    df_result_gender.columns = ['CONTENT','COEF','PVALUE']
    df_result_gender['GENDER'] = df_result_gender['CONTENT']
    df_result_gender['GENDER'] = df_result_gender['GENDER'].str.replace("[","")
    df_result_gender['GENDER'] = df_result_gender['GENDER'].str.replace("]","")
    df_result_gender['GENDER'] = df_result_gender['GENDER'].str.split('.').str[-1]
    df_result_gender = df_result_gender.drop(columns=['CONTENT'])
    df_result_gender['COEF_DISPLAY'] = df_result_gender['COEF']
    df_result_gender['STAT'] = "No"
    df_result_gender['STAT_COUNT'] = 0
    df_result_gender.loc[df_result_gender['PVALUE']<=0.05,"STAT"] = "Yes"
    df_result_gender.loc[((df_result_gender['PVALUE']<=0.05) & (df_result_gender['COEF']<0)) ,"STAT_COUNT"] = 1
    df_result_gender = df_result_gender.sort_values(by=['COEF_DISPLAY'], ascending=False)
    baseline_row = pd.DataFrame({'GENDER':'M', 'COEF':0, 'COEF_DISPLAY':0,'PVALUE':1,'STAT':"No",'STAT_COUNT':0},index =[0])
    df_result_gender = pd.concat([baseline_row, df_result_gender[:]]).reset_index(drop = True)
    df_result_gender['GENDER'] = df_result_gender.apply(lambda x: gender_name_replace(text=x['GENDER']),axis=1)
    df_result_gender = df_result_gender[~((df_result_gender['GENDER']=='Unknown') | (df_result_gender['GENDER']=='unknown'))]
    df_result_gender = df_result_gender.reset_index(drop=True)
    df_result_gender.to_excel('result_gender.xlsx')
    fig_gender_bar = plot_bar(df_result_gender,'GENDER')
    
    # Result Ethnicity Table
    df_result_eth = df_result[df_result['CONTENT'].str.contains("ETHNICITY")==True]
    df_result_eth = df_result_eth[['CONTENT','Coef.','P>|t|']]
    df_result_eth.columns = ['CONTENT','COEF','PVALUE']
    df_result_eth['ETHNICITY'] = df_result_eth['CONTENT']
    df_result_eth['COEF_DISPLAY'] = df_result_eth['COEF']
    df_result_eth['STAT'] = "No"
    df_result_eth['STAT_COUNT'] = 0
    df_result_eth.loc[df_result_eth['PVALUE']<=0.05,"STAT"] = "Yes"
    df_result_eth.loc[((df_result_eth['PVALUE']<=0.05) & (df_result_eth['COEF']<0)),"STAT_COUNT"] = 1
    df_result_eth = df_result_eth.sort_values(by=['COEF_DISPLAY'], ascending=False)
    df_result_eth['ETHNICITY'] = df_result_eth['ETHNICITY'].str.replace("[","")
    df_result_eth['ETHNICITY'] = df_result_eth['ETHNICITY'].str.replace("]","")
    df_result_eth['ETHNICITY'] = df_result_eth['ETHNICITY'].str.split('.').str[-1]
    df_result_eth = df_result_eth.drop(columns=['CONTENT'])
    df_result_eth = df_result_eth.reset_index(drop=True)
    baseline_eth = df['ETHNICITY'].value_counts()
    # baseline_eth.to_excel('baseline_eth.xlsx')
    # print(baseline_eth.index[0])
    baseline_row = pd.DataFrame({'ETHNICITY':baseline_eth.index[0], 'COEF':0, 'COEF_DISPLAY':0,'PVALUE':1,'STAT':"No",'STAT_COUNT':0},index =[0])
    df_result_eth = pd.concat([baseline_row, df_result_eth[:]]).reset_index(drop = True)
    df_result_eth = df_result_eth[~((df_result_eth['ETHNICITY']=='Unknown') | (df_result_eth['ETHNICITY']=='unknown'))]
    eth_baseline = df_result_eth['ETHNICITY'][0]
    df_result_eth.to_excel('result_eth.xlsx')
    fig_eth_bar = plot_bar(df_result_eth,'ETHNICITY')
    
    # Result All Table
    df_initial_result = process_run_result(results)
    df_initial_result.to_excel('initial_result.xlsx')
    
    # Gender Table
    df_gender = df.pivot_table(index=['GENDER'],values=['EEID','SALARY'],aggfunc={'EEID':'count','SALARY':'mean'},fill_value=np.nan)
    df_gender.columns = ['_'.join(col).strip() for col in df_gender.columns.values]
    df_gender = df_gender.reset_index()
    df_gender.columns = ['GENDER','HC','AVG_PAY']
    df_gender = df_gender.merge(df_gender_name,on='GENDER',how='left')
    df_gender = df_gender.sort_values(by=['HC'],ascending=False)
    
    df_gender_female_pay = df[df['GENDER'] == 'F']
    df_gender_nonb_pay = df[df['GENDER'] == 'N']
    
    gender_female_pay = df_gender_female_pay['SALARY'].mean()
    gender_nonb_pay = df_gender_nonb_pay['SALARY'].mean()
    
    # df_gender_minor_pay.to_excel('gender.xlsx')
    fig_gender_hc = plot_gender_donut(df_gender)
    
    # Ethnicity Table
    # plt.clf()
    df_eth = None
    eth_minor_pay = None
    if 'ETHNICITY' in include_feature:
        df_eth = df.pivot_table(index=['ETHNICITY'],values=['EEID','SALARY'],aggfunc={'EEID':'count','SALARY':'mean'},fill_value=np.nan)
        df_eth.columns = ['_'.join(col).strip() for col in df_eth.columns.values]
        df_eth = df_eth.reset_index()
        df_eth.columns = ['ETHNICITY_NAME','HC','AVG_PAY']
        df_eth = df_eth.sort_values(by=['HC'],ascending=False)
        fig_eth_hc = plot_eth_donut(data = df_eth)
        df_eth_minor_pay = df[df['ETHNICITY'] != df_eth['ETHNICITY_NAME'][0]]
        eth_minor_pay = df_eth_minor_pay['SALARY'].mean()
        # df_eth_minor_pay.to_excel('eth.xlsx')
    
    # df_gender = df_gender.drop(columns=['GENDER'])
    # df_gender.to_excel('gender.xlsx')
    # df.to_excel('gender.xlsx')
    # print(message.loc[['OVERVIEW']]['Message'])
    
    return df, df_org, df_validation, message, exclude_col, r2_raw, female_coff_raw, female_pvalue_raw, r2, female_coff, female_pvalue, before_clean_record, after_clean_record,hc_female,X_full,budget_df,exclude_feature, include_feature,df_gender, df_eth, fig_gender_hc,fig_eth_hc, avg_pay, gender_female_pay, gender_nonb_pay, eth_minor_pay,fig_gender_bar, fig_eth_bar, df_result_gender, df_result_eth,eth_baseline, df_initial_result, predict_df, ci_view_lower, ci_view_upper

def display_rename(display_map,feature):
    return [display_map.get(item,item)  for item in feature]

# def analysis(df_submit, run_demo, file_path, display_path, main_page, main_page_info, ci):
def analysis(df_submit, run_demo, file_path, display_path, main_page, main_page_info):
    # Process df (not demo datafile)    
    # with st.spinner('Running analysis, Please wait for it...'):
    m_info = main_page_info.success('Reading Data')
    if run_demo == True:
        # Demo Run
        df = pd.read_excel(file_path,sheet_name="Demo Data")
    else:
        df = pd.read_excel(df_submit,sheet_name="Submission")    

    # Convert program column names to display
    df_name = pd.read_excel(display_path,sheet_name="Standard_Col")
    df_cus_name = pd.read_excel(display_path,sheet_name="Custom_Col")
    df_gender_name = pd.read_excel(display_path,sheet_name="Gender")

    # Run Checking    
    df_check = df.iloc[1:]
    full_list = df_name['PROGRAM_NAME'].tolist()
    full_list_display = df_name['DISPLAY_NAME'].tolist()

    req_list = df_name[df_name['REQUIRE']==1]['PROGRAM_NAME'].tolist()
    # req_list_display = df_name[df_name['REQUIRE']==1]['DISPLAY_NAME'].tolist()
    cus_list = df_cus_name['PROGRAM_NAME'].tolist()
    display_map = dict(zip(full_list, full_list_display))
    display_inv_map = dict(zip(full_list_display, full_list))

    inc_list = []
    exc_list = []

    for col in df_check.columns.tolist():
        if df_check[col].replace(r'^\s*$', np.nan, regex=True).isna().all():
        # if df_check[col].isna():
            exc_list.append(col)
        else:
            inc_list.append(col)

    error_req_list = [x for x in req_list if x not in inc_list]
    error_req_message=""
    if len(error_req_list) != 0:
        error_req_message = 'The requested data field below has not been provided: '+ ', '.join(error_req_list)+'. Please update and resubmit the template.'
    
    final_req_list = [x for x in req_list if x not in ['EEID','SALARY']]
    display_req_list = display_rename(display_map,final_req_list)

    final_optional_list = [x for x in inc_list if x not in (req_list+cus_list+['SNAPSHOT_DATE'])]
    display_optional_list = display_rename(display_map,final_optional_list)

    # Display selection Step 3
    config = main_page.form("Configuration")
    with config:
        config.write("Step 3. Confirm Selected Configuration")
        ci_select, optional_select, req_select, = config.columns((0.5, 1, 1)) 
        ci = ci_select.slider(label = 'A: Choose fair pay confidence internal %', value = 95, min_value = 70, max_value = 99, step = 1, help='Setting at 95% means I want to have a pay range so that 95% of the time, the predicted pay falls inside.')
        ci = ci/100
        optional_col = optional_select.multiselect(label = 'C: Select Optional Pay Factors',options=display_optional_list,default=display_optional_list,disabled=False)
        req_col = req_select.multiselect(label = 'B: Required Pay Factors',options=display_req_list,default=display_req_list,disabled=True)
        submitted_form = config.form_submit_button("🚀 Confirm to Run Analysis'")

    # Final Filter on pay driver selection
    optional_col_select = display_rename(display_inv_map,optional_col)
    req_list = ['SNAPSHOT_DATE'] + req_list
    final_col_select = req_list + optional_col_select
    df =  df[final_col_select]
    
    print('inc: ')
    print(inc_list)
    print('exc: ')
    print(exc_list)
    print('req: ')
    print(req_list)
    print('cus: ')
    print(cus_list)
    print('error message req: ')
    print(error_req_message)
    print('display req')
    print(display_req_list)
    print('display optional')
    print(display_optional_list)
    print('display system run optional')
    print(optional_col_select)

    # st.stop()

    if submitted_form:
        # st.write(optional_col_select)
        # st.write(final_col_select)
        # st.write(df.columns.tolist())
        display_map = dict(zip(df_name['PROGRAM_NAME'], df_name['DISPLAY_NAME']))
        
        # Run discovery model:
        m_info = main_page_info.success('Running Gap Analysis')
        df, df_org,  df_validation, message, exclude_col, r2_raw, female_coff_raw, female_pvalue_raw, r2, female_coff, female_pvalue, before_clean_record, after_clean_record,hc_female,X_full,budget_df,exclude_feature, include_feature,df_gender,df_eth,fig_gender_hc,fig_eth_hc,avg_pay, gender_female_pay, gender_nonb_pay, eth_minor_pay,fig_gender_bar, fig_eth_bar,df_result_gender, df_result_eth, eth_baseline, df_initial_result, predict_df, ci_view_lower, ci_view_upper = run(df,df_gender_name,req_list,ci)     
        
        # st.write(df.columns.tolist())
        # , fig_gender_bar
        # ,fig_eth_hc
        print('pvalue'+str(female_pvalue))
        gender_gap_format = str(round(female_coff*100,0))+'%'
        gender_gap_stats = '**NOT** statistically significant'
        if female_pvalue<0.05:
            gender_gap_stats = 'statistically significant'

        # Run Reme Pvalue = 7%
        
        df_result = copy.deepcopy(df_initial_result)
        count_loop = 0
        count_reme = 0
        reme_exit = 0
        seek_budget_df_pv = copy.deepcopy(budget_df)
        
        while df_result['STAT_COUNT'].sum()>0:
            # print('count_loop: '+str(count_loop))
            df_result_loop = zip(df_result['CONTENT'].to_list(), df_result['FEATURE'].to_list(), df_result['COEF'].to_list(), df_result['PVALUE'].to_list())
            
            for protect_group_class, project_group_feature,init_current_gap,init_current_pvalue in df_result_loop:
                # print('count_reme: '+str(count_reme))
                m_info = main_page_info.success('Running Remediation Scenario B: Mitigate Legal Risk - '+ project_group_feature + ' - '+ protect_group_class)
                if count_reme == 0:
                    current_gap = init_current_gap
                    current_pvalue = init_current_pvalue
                else:
                    current_gap = df_result[df_result['CONTENT']==protect_group_class]['COEF'].values[0]
                    current_pvalue = df_result[df_result['CONTENT']==protect_group_class]['PVALUE'].values[0]

                seek_budget_df_pv,seek_budget_pv,seek_resulting_gap_pv,seek_resulting_pvalues_pv,seek_adj_count_pv, seek_adj_budget_pct_pv,seek_pass_pv, seek_success_pv, df_result = reme_pvalue_seek(df,seek_budget_df_pv,X_full, project_group_feature=project_group_feature, protect_group_class=protect_group_class, seek_goal=0.07, current_gap = current_gap, current_pvalue = current_pvalue, input_df_result = df_result, count_loop = count_loop, search_step = -0.001)
                
                pv_budget_name = 'pv_budget_'+str(count_loop)+'_'+protect_group_class+'.xlsx'
                pv_result_name = 'pv_result_'+str(count_loop)+'_'+protect_group_class+'.xlsx'
                
                seek_budget_df_pv.to_excel(pv_budget_name)
                df_result.to_excel(pv_result_name)
                
                count_reme = count_reme + 1
                print('count_reme: '+str(count_reme))
                if df_result['STAT_COUNT'].sum()==0:
                    reme_exit = 1
                    break
            if reme_exit == 1:
                break
            count_loop = count_loop +1
            print('count_loop: '+str(count_loop))
        
#         df_result_loop = zip(df_initial_result['CONTENT'].to_list(), df_initial_result['FEATURE'].to_list(), df_initial_result['COEF'].to_list(), df_initial_result['PVALUE'].to_list())        
#         seek_budget_df_pv = copy.deepcopy(budget_df)
        
#         count_reme = 0
        
#         for protect_group_class, project_group_feature,init_current_gap,init_current_pvalue in df_result_loop:
#             m_info = main_page_info.success('Running Remediation Scenario B: Mitigate Legal Risk - '+ protect_group_class)
#             if count_reme == 0:
#                 current_gap = init_current_gap
#                 current_pvalue = init_current_pvalue
#             else:
#                 current_gap = df_result[df_result['']==]['COEF'].values[0]
#                 current_pvalue = df_result[df_result['']==]['PVALUE'].values[0]
            
#             seek_budget_df_pv,seek_budget_pv,seek_resulting_gap_pv,seek_resulting_pvalues_pv,seek_adj_count_pv, seek_adj_budget_pct_pv,seek_pass_pv, seek_success_pv, df_result = reme_pvalue_seek(df,seek_budget_df_pv,X_full, project_group_feature=project_group_feature, protect_group_class=protect_group_class, seek_goal=0.07, current_gap = current_gap, current_pvalue = current_pvalue, search_step = -0.001)
            
            
            # seek_budget_df_pv.to_excel('df_pv_n.xlsx')
            # df_result.to_excel('df_pv_n_result.xlsx')
        
        asdf
        
        # for 
            
#         # Run Reme Pvalue = 7%
#         m_info = main_page_info.success('Running Remediation Scenario A: Mitigate Legal Risk')
#         seek_budget_df_pv,seek_budget_pv,seek_resulting_gap_pv,seek_resulting_pvalues_pv,seek_adj_count_pv, seek_adj_budget_pct_pv,seek_pass_pv, seek_success_pv = reme_pvalue_seek(df,budget_df,X_full, project_group_feature='GENDER', protect_group_class='F', seek_goal=0.07, current_gap = female_coff, current_pvalue = female_pvalue, search_step = -0.001)
#         seek_budget_df_pv.to_excel('df_pv.xlsx')

#         # Run Reme Zero Gap
#         m_info = main_page_info.success('Running Remediation Scenario B: Close Gender Gap')
#         seek_budget_df_gap,seek_budget_gap,seek_resulting_gap_gap,seek_resulting_pvalues_gap,seek_adj_count_gap, seek_adj_budget_pct_gap,seek_pass_gap, seek_success_gap = reme_gap_seek(df,budget_df,X_full, project_group_feature='GENDER', protect_group_class='F', seek_goal=0, current_gap = female_coff, current_pvalue = female_pvalue, search_step = -0.001) 
#         seek_budget_df_gap.to_excel('df_gap.xlsx')
        
#         # Run Reme Pvalue = 7%
#         m_info = main_page_info.success('Running Remediation Scenario A: Mitigate Legal Risk')
#         seek_budget_df_pv,seek_budget_pv,seek_resulting_gap_pv,seek_resulting_pvalues_pv,seek_adj_count_pv, seek_adj_budget_pct_pv,seek_pass_pv, seek_success_pv = reme_pvalue_seek(df,budget_df,X_full, project_group_feature='GENDER', protect_group_class='N', seek_goal=0.07, current_gap = female_coff, current_pvalue = female_pvalue, search_step = -0.001)
#         seek_budget_df_pv.to_excel('df_pv.xlsx')

#         # Run Reme Zero Gap
#         m_info = main_page_info.success('Running Remediation Scenario B: Close Gender Gap')
#         seek_budget_df_gap,seek_budget_gap,seek_resulting_gap_gap,seek_resulting_pvalues_gap,seek_adj_count_gap, seek_adj_budget_pct_gap,seek_pass_gap, seek_success_gap = reme_gap_seek(df,budget_df,X_full, project_group_feature='GENDER', protect_group_class='N', seek_goal=0, current_gap = female_coff, current_pvalue = female_pvalue, search_step = -0.001) 
#         seek_budget_df_gap.to_excel('df_gap.xlsx')
        
        # Run Reme Pvalue = 7%
        
        m_info = main_page_info.success('Running Remediation Scenario B: Mitigate Legal Risk')
        seek_budget_df_pv,seek_budget_pv,seek_resulting_gap_pv,seek_resulting_pvalues_pv,seek_adj_count_pv, seek_adj_budget_pct_pv,seek_pass_pv, seek_success_pv, df_result = reme_pvalue_seek(df,budget_df,X_full, project_group_feature='GENDER', protect_group_class='N', seek_goal=0.07, current_gap = female_coff, current_pvalue = female_pvalue, search_step = -0.001)
        seek_budget_df_pv.to_excel('df_pv_n.xlsx')
        df_result.to_excel('df_pv_n_result.xlsx')
        
        asdf
        
        m_info = main_page_info.success('Running Remediation Scenario B: Mitigate Legal Risk')
        seek_budget_df_pv,seek_budget_pv,seek_resulting_gap_pv,seek_resulting_pvalues_pv,seek_adj_count_pv, seek_adj_budget_pct_pv,seek_pass_pv, seek_success_pv, df_result = reme_pvalue_seek(df,seek_budget_df_pv,X_full, project_group_feature='GENDER', protect_group_class='F', seek_goal=0.07, current_gap = female_coff, current_pvalue = female_pvalue, search_step = -0.001)
        seek_budget_df_pv.to_excel('df_pv_female.xlsx')
        df_result.to_excel('df_pv_female_result.xlsx')
        
        m_info = main_page_info.success('Running Remediation Scenario B: Mitigate Legal Risk')
        seek_budget_df_pv,seek_budget_pv,seek_resulting_gap_pv,seek_resulting_pvalues_pv,seek_adj_count_pv, seek_adj_budget_pct_pv,seek_pass_pv, seek_success_pv, df_result = reme_pvalue_seek(df,seek_budget_df_pv,X_full, project_group_feature='ETHNICITY', protect_group_class='Black', seek_goal=0.07, current_gap = female_coff, current_pvalue = female_pvalue, search_step = -0.001)
        seek_budget_df_pv.to_excel('df_pv_black.xlsx')
        df_result.to_excel('df_pv_black_result.xlsx')
        
        # asdf
        
        m_info = main_page_info.success('Running Remediation Scenario B: Mitigate Legal Risk')
        seek_budget_df_pv,seek_budget_pv,seek_resulting_gap_pv,seek_resulting_pvalues_pv,seek_adj_count_pv, seek_adj_budget_pct_pv,seek_pass_pv, seek_success_pv, df_result = reme_pvalue_seek(df,seek_budget_df_pv,X_full, project_group_feature='ETHNICITY', protect_group_class='Asian', seek_goal=0.07, current_gap = female_coff, current_pvalue = female_pvalue, search_step = -0.001)
        seek_budget_df_pv.to_excel('df_pv_asian.xlsx')
        df_result.to_excel('df_pv_asian_result.xlsx')
        
        m_info = main_page_info.success('Running Remediation Scenario B: Mitigate Legal Risk')
        seek_budget_df_pv,seek_budget_pv,seek_resulting_gap_pv,seek_resulting_pvalues_pv,seek_adj_count_pv, seek_adj_budget_pct_pv,seek_pass_pv, seek_success_pv, df_result = reme_pvalue_seek(df,seek_budget_df_pv,X_full, project_group_feature='GENDER', protect_group_class='F', seek_goal=0.07, current_gap = female_coff, current_pvalue = female_pvalue, search_step = -0.001)
        seek_budget_df_pv.to_excel('df_pv_female_2.xlsx')
        df_result.to_excel('df_pv_female_2_result.xlsx')
        
        m_info = main_page_info.success('Running Remediation Scenario B: Mitigate Legal Risk')
        seek_budget_df_pv,seek_budget_pv,seek_resulting_gap_pv,seek_resulting_pvalues_pv,seek_adj_count_pv, seek_adj_budget_pct_pv,seek_pass_pv, seek_success_pv, df_result = reme_pvalue_seek(df,seek_budget_df_pv,X_full, project_group_feature='ETHNICITY', protect_group_class='Black', seek_goal=0.07, current_gap = female_coff, current_pvalue = female_pvalue, search_step = -0.001)
        seek_budget_df_pv.to_excel('df_pv_black2.xlsx')
        df_result.to_excel('df_pv_black2_result.xlsx')
        
        # asdf
        
        m_info = main_page_info.success('Running Remediation Scenario B: Mitigate Legal Risk')
        seek_budget_df_pv,seek_budget_pv,seek_resulting_gap_pv,seek_resulting_pvalues_pv,seek_adj_count_pv, seek_adj_budget_pct_pv,seek_pass_pv, seek_success_pv, df_result = reme_pvalue_seek(df,seek_budget_df_pv,X_full, project_group_feature='ETHNICITY', protect_group_class='Asian', seek_goal=0.07, current_gap = female_coff, current_pvalue = female_pvalue, search_step = -0.001)
        seek_budget_df_pv.to_excel('df_pv_asian2.xlsx')
        df_result.to_excel('df_pv_asian2_result.xlsx')
        
        asdf

        # Run Reme Zero Gap
        m_info = main_page_info.success('Running Remediation Scenario C: Close Gender Gap')
        seek_budget_df_gap,seek_budget_gap,seek_resulting_gap_gap,seek_resulting_pvalues_gap,seek_adj_count_gap, seek_adj_budget_pct_gap,seek_pass_gap, seek_success_gap = reme_gap_seek(df,budget_df,X_full, project_group_feature='ETHNICITY', protect_group_class='Black', seek_goal=0, current_gap = female_coff, current_pvalue = female_pvalue, search_step = -0.001) 
        seek_budget_df_gap.to_excel('df_gap.xlsx')

        # Create download file for remediation
        reme_download_flag = 0
        df_reme_org = df.drop(columns=['VALIDATION_MESSAGE', 'VALIDATION_FLAG', 'NOW','LOG_SALARY'])
        if operator.not_(seek_budget_df_pv.empty):
            df_reme_ind = seek_budget_df_pv.merge(seek_budget_df_gap,on='EEID',how='inner').merge(df_reme_org,on = 'EEID',how='inner')
            list_reme = [x for x in df_reme_ind.columns.tolist() if x not in ['EEID','GENDER','SALARY']]
            list_reme = ['EEID','GENDER','SALARY']+list_reme
            df_reme_ind = df_reme_ind[list_reme]
            reme_download_flag = 1
        elif (seek_budget_df_pv.empty) and (operator.not_(seek_budget_df_gap.empty)):
            df_reme_ind = seek_budget_df_gap.merge(df_reme_org,on = 'EEID',how='inner')
            list_reme = [x for x in df_reme_ind.columns.tolist() if x not in ['EEID','GENDER','SALARY']]
            list_reme = ['EEID','GENDER','SALARY']+list_reme
            df_reme_ind = df_reme_ind[list_reme]
            reme_download_flag = 1
        # df_reme_ind.to_excel('df_reme_ind.xlsx')

        # Run Remediation Messages
        message_budget_pv = np.nan
        if seek_pass_pv == False:
            message_budget_pv = '0 - gender pay gap is currently not statistically significant.'
            message_budget_pv_overview = message_budget_pv
        elif (seek_pass_pv == True) and (seek_success_pv == False):
            message_budget_pv = 'No results found, please contact our consultant for more information.'
            message_budget_pv_overview = message_budget_pv
        else:
            if seek_budget_pv> 1000000:
                message_budget_pv = str(locale.format("%.2f", round(seek_budget_pv/1000000,2), grouping=True))+' Million'+'\n'+'('+str(round(seek_adj_budget_pct_pv*100,0))+'% of Pay)'
            else:
                message_budget_pv = str(locale.format("%d", round(seek_budget_pv/1000,0), grouping=True))+' K'+'\n'+'('+str(round(seek_adj_budget_pct_pv*100,0))+'% of Pay)'
            message_budget_pv_overview = "Raising women's pay by $"+message_budget_pv+' will reduce gap to ' + '-2.2%' + ' and become statistically insignificant.'

        message_budget_gap = np.nan
        if seek_pass_gap == False:
            message_budget_gap = '0 - Women earn more than men, so no adjustment is necessary.'
        elif (seek_pass_gap == True) and (seek_success_gap == False):
            message_budget_gap = 'No results found, please contact our consultant for more information.'
        else:
            if seek_budget_gap> 1000000:
                message_budget_gap = str(locale.format("%.2f", round(seek_budget_gap/1000000,2), grouping=True))+' Million'+'\n'+'('+str(round(seek_adj_budget_pct_gap*100,2))+'% of Pay)'
            else:
                message_budget_gap = str(locale.format("%d", round(seek_budget_gap/1000,0), grouping=True))+' K'+'\n'+'('+str(round(seek_adj_budget_pct_gap*100,2))+'% of Pay)'

        scenario = ['Current','A','B']
        action = ['🏁 No change','✔️ Mitigate legal risk \n'+'✔️ Reduce the gender gap to a statistical insignificant level.','✔️ Mitigate legal risk \n'+'✔️ Completely close gender gap \n'+'✔️ Become a market leader (Top 1%)\n']
        budget = ['0',message_budget_pv,message_budget_gap]

        if abs(seek_resulting_gap_gap)<0.0005:
                seek_resulting_gap_gap = 0
        net_gap = [female_coff,seek_resulting_gap_pv,seek_resulting_gap_gap]
        net_gap = [f'{i*100:.1f}%' for i in net_gap]

        df_reme = pd.DataFrame({'Scenario': scenario, 'How do I do this?': action, 'What is my budget?': budget, 'What is the gap after adjustment?': net_gap})

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
        styler = df_reme.style.hide_index().set_table_styles([cell_hover, index_names, headers], overwrite=False).set_properties(**{
    'white-space': 'pre-wrap'})
        
        # Show exclude and include features
        include_feature = [display_map.get(item,item)  for item in include_feature]
        exclude_feature = [display_map.get(item,item)  for item in exclude_feature]

        include_feature_text =  ', '.join(include_feature)
        exclude_feature_text =  ', '.join(exclude_feature)
        # st.sidebar.options = st.sidebar.markdown('Pay drivers included in model:' + include_feature_text)

        # Run data validation
        m_info = main_page_info.success('Output Data Validation')
        demo_validation = convert_df(df_validation)

        output = BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        df_validation.to_excel(writer, index=False, sheet_name='Sheet1')
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']

        for column in df_validation:
            column_width = max(df_validation[column].astype(str).map(len).max(), len(column))+3
            col_idx = df_validation.columns.get_loc(column)
            writer.sheets['Sheet1'].set_column(col_idx, col_idx, column_width)
        cell_format = workbook.add_format()

        cell_format.set_pattern(1)  # This is optional when using a solid fill.
        cell_format.set_bg_color('yellow')
        worksheet.write('A1', 'Please see the instructions for a valid record.',cell_format)

        writer.save()
        processed_data = output.getvalue()

# Start Streamlit ----------------------------------------------------------------------------------------------------------------------
        
        # Display run is successful message    
        m_info = main_page_info.success('View Result: '+message.loc[['OVERVIEW']][0])

        main_page.markdown("""---""")
        m_col1_but_col1, m_col1_but_col2, m_col1_but_col3, m_col1_but_col4 = main_page.columns((0.5, 0.5, 1 , 1))

        m_col1_but_col1.metric('Submitted Entry',before_clean_record)
        m_col1_but_col1.metric('Processed Entry',after_clean_record)
        m_col1_but_col1.metric('Invalid Entry',before_clean_record - after_clean_record)
        if operator.not_(df_validation.empty):
            m_col1_but_col1.markdown(get_excel_file_downloader_html(processed_data, 'Invalid Entry.xlsx'), unsafe_allow_html=True)
            m_col1_but_col1.markdown("🖱️ 'Save link as...'")

        avg_pay_info = str(locale.format("%d", round(avg_pay/1000,0), grouping=True))+'k'
        m_col1_but_col2.metric('Ave Overall Salary',avg_pay_info)

        gender_female_pay_info = str(locale.format("%d", round(gender_female_pay/1000,0), grouping=True))+'k'
        gender_female_pay_delta = str(locale.format("%d", round(gender_female_pay/1000,0) - round(avg_pay/1000,0), grouping=True))+'k'
        # m_col1_but_col2.metric('Avg Non-Male Salary',gender_minor_pay_info, delta=gender_minor_pay_delta)
        m_col1_but_col2.metric('Avg Female Salary',gender_female_pay_info)

        gender_nonb_pay_info = str(locale.format("%d", round(gender_nonb_pay/1000,0), grouping=True))+'k'
        gender_nonb_pay_delta = str(locale.format("%d", round(gender_nonb_pay/1000,0) - round(avg_pay/1000,0), grouping=True))+'k'
        # m_col1_but_col2.metric('Avg Non-Male Salary',gender_minor_pay_info, delta=gender_minor_pay_delta)
        m_col1_but_col2.metric('Avg Non-Binary Salary',gender_nonb_pay_info)

        eth_major_name = 'Avg Non-'+ df_eth['ETHNICITY_NAME'][0] + ' Salary'
        eth_minor_pay_info = str(locale.format("%d", round(eth_minor_pay/1000,0), grouping=True))+'k'
        eth_minor_pay_delta = str(locale.format("%d", round(eth_minor_pay/1000,0) - round(avg_pay/1000,0), grouping=True))+'k'
        # m_col1_but_col2.metric(eth_major_name,eth_minor_pay_info, delta = eth_minor_pay_delta)
        m_col1_but_col2.metric(eth_major_name,eth_minor_pay_info)

        # m_col1_but_col1_2.metric('Submission Record',before_clean_record)
        # m_col1_but_col1_2.metric('Successful Run',after_clean_record)

        m_col1_but_col3.markdown("<h1 style='text-align: center; vertical-align: top; font-size: 100%'>Gender Distribution</h1>", unsafe_allow_html=True)
        m_col1_but_col3.pyplot(fig_gender_hc)

        m_col1_but_col4.markdown("<h1 style='text-align: center; vertical-align: top; font-size: 100%'>Ethnicity Distribution</h1>", unsafe_allow_html=True)
        m_col1_but_col4.pyplot(fig_eth_hc)

        # main_page.markdown("""---""")

        # main_page.pyplot(fig_gender_hc)
#         test1, test2, test3 = main_page.columns((1, 1, 1))

#         test1.plotly_chart(fig_gender_bar, use_container_width=True)

        # main_page.markdown("""---""")

        # with st.expander("What pay drivers are supplied:"):
        #     inc_col, exc_col = st.columns((1, 1))
        #     inc_col.markdown("<h1 style='text-align: left; vertical-align: bottom;color: Green; font-size: 110%; opacity: 0.7'> ✔️ Pay drivers you supplied:  </h1>", unsafe_allow_html=True)
        #     inc_col.markdown(include_feature_text, unsafe_allow_html=True)       
        #     exc_col.markdown("<h1 style='text-align: left; vertical-align: bottom;color: Orange; font-size: 110%; opacity: 0.7'> ⚠️ Pay drivers you haven't supplied:  </h1>", unsafe_allow_html=True)        
        #     exc_col.markdown(exclude_feature_text, unsafe_allow_html=True)

        # r2= 0.9

        # Show R2
        main_page.markdown("""---""")
        metric_R2_1, metric_R2_2, metric_R2_3 = main_page.columns((1, 1.2, 1.3))            
        metric_R2_1.markdown("<h1 style='text-align: left; vertical-align: bottom; font-size: 150%; color: #3498DB; opacity: 0.7'>Robustness</h1>", unsafe_allow_html=True)
        # metric_R2_1.plotly_chart(fig_r2_gender_gap, use_container_width=True)
        with metric_R2_1:
            r2_options = get_r2_option(r2)
            st_echarts(options=r2_options,height="200px") 

        metric_R2_2.markdown("<h1 style='text-align: left; vertical-align: bottom;color: #3498DB; font-size: 150%; opacity: 0.7'>Benchmark</h1>", unsafe_allow_html=True)
        metric_R2_2.markdown("<h1 style='text-align: left; vertical-align: bottom;color: Green; font-size: 110%; opacity: 0.7'> 🌐 70% ~ 100%  </h1>" "  \n"  "Robustness measures whether the standard model adequately explains compensation decisions. For instance, 80% means that the standard model explains 80% of the pay difference between employees.", unsafe_allow_html=True)
        metric_R2_3.markdown("<h1 style='text-align: left; vertical-align: bottom;color: #3498DB; font-size: 150%; opacity: 0.7'>Observation</h1>", unsafe_allow_html=True)
        if r2>=0.7:
            metric_R2_3.markdown("<h1 style='text-align: left; vertical-align: bottom;color: Green; font-size: 110%; opacity: 0.7'> ✔️ Align with market  </h1>", unsafe_allow_html=True)
            metric_R2_3.markdown("Great! Pay drivers are **sufficient** to account for the variation of wages between employees. Let us look at the pay gap.", unsafe_allow_html=True)
        else:
            metric_R2_3.markdown("<h1 style='text-align: left; vertical-align: bottom;color: Orange; font-size: 110%; opacity: 0.7'> ⚠️ Below market  </h1>" , unsafe_allow_html=True)
            metric_R2_3.markdown("The default compensation drivers are <font color=Orange> **not robust** </font> in drawing conclusions on the salary gap. In general, we can improve the model robustness by adding additional drivers such as talent potential, cost centre, skills and so on. Please contact us for an open consultation.", unsafe_allow_html=True)
            st.stop()
        # metric_R2_3.markdown("<h1 style='text-align: center; vertical-align: bottom; color: Black; background-color: #3498DB; opacity: 0.7; border-style: dotted'>Observation</h1>", unsafe_allow_html=True)

        # Show Gender Gap
        main_page.markdown("""---""")
        metric_net_gap_1, metric_net_gap_2, metric_net_gap_3 = main_page.columns((1, 1.1, 1.4))            
        metric_net_gap_1.markdown("<h1 style='text-align: left; vertical-align: bottom; font-size: 150%; color: #3498DB; opacity: 0.7'> Gender Gap </h1>", unsafe_allow_html=True)
        metric_net_gap_1.plotly_chart(fig_gender_bar, use_container_width=True)

        metric_net_gap_2.markdown("<h1 style='text-align: left; vertical-align: bottom;color: #3498DB; font-size: 150%; opacity: 0.7'>Benchmark</h1>", unsafe_allow_html=True)
        metric_net_gap_2.write("<h1 style='text-align: left; vertical-align: bottom;color: Green; font-size: 110%; opacity: 0.7'> 🌐 -3% ~ 1% </h1>" "Pay gap measures for every dollar paid to male employees, how much (less) or more goes to non-male employees. For example pay gap at -10% means that on average women are paid 10% less compared to men all else equal. In US, gender gap typically ranges between -3% and +1%. ", unsafe_allow_html=True)    

        num_gender_sig = df_result_gender['STAT_COUNT'].sum()
        # num_gender_sig = 0
        metric_net_gap_3.markdown("<h1 style='text-align: left; vertical-align: bottom;color: #3498DB; font-size: 150%; opacity: 0.7'>Observation</h1>", unsafe_allow_html=True)
        if num_gender_sig>0:
            metric_net_gap_3.markdown("<h1 style='text-align: left; vertical-align: bottom;color: Orange; font-size: 110%; opacity: 0.7'> ⚠️ Legal Risk - High </h1>", unsafe_allow_html=True)
            metric_net_gap_3.write('There is/are '+ str(num_gender_sig) + ' statistically significant gender pay gap shown with ' + '<font color=Orange> **Red** </font>' +' bar. A significant gender pay gap means - we are more than 95% certain that the gap exists after incorporates all of the legitimate determinants of pay (such as differences of skill, effort, and responsibility). From a legal perspective:',unsafe_allow_html=True) 
            metric_net_gap_3.write('* Statistically significant gap - Strong evidence of gender pay discrimination' +'\n'+'* Non statistically significant gap - No evidence of gender pay discrimination, the gap is likely due to random chance',unsafe_allow_html=True)
            metric_net_gap_3.write('You may consider reducing the pay gap to a statistically insignificant level to reduce legal risk.')
        else:
            metric_net_gap_3.markdown("<h1 style='text-align: left; vertical-align: bottom;color: Green; font-size: 110%; opacity: 0.7'> ✔️ Legal Risk - Low </h1>", unsafe_allow_html=True)
            metric_net_gap_3.write('There is no negative statistically significant gender pay gap shown in the chart. A significant gender pay gap means - we are more than 95% certain that the gap exists after incorporates all of the legitimate determinants of pay (such as differences of skill, effort, and responsibility). From a legal perspective:',unsafe_allow_html=True) 
            metric_net_gap_3.write('* Statistically significant gap - Strong evidence of gender pay discrimination' +'\n'+'* Non statistically significant gap - No evidence of gender pay discrimination, the gap is likely due to random chance',unsafe_allow_html=True)
            metric_net_gap_3.write('As a precaution, you can routinely repeat this analysis to monitor the pay gap. An alternative is to consider completely closing the pay gap to zero.')                

        # Show Ethnicity Gap
        main_page.markdown("""---""")
        metric_eth_gap_1, metric_eth_gap_2, metric_eth_gap_3 = main_page.columns((1, 1.1, 1.4))            
        metric_eth_gap_1.markdown("<h1 style='text-align: left; vertical-align: bottom; font-size: 150%; color: #3498DB; opacity: 0.7'> Ethnicity Gap </h1>", unsafe_allow_html=True)
        metric_eth_gap_1.plotly_chart(fig_eth_bar, use_container_width=True)

        metric_eth_gap_2.markdown("<h1 style='text-align: left; vertical-align: bottom;color: #3498DB; font-size: 150%; opacity: 0.7'>Benchmark</h1>", unsafe_allow_html=True)
        metric_eth_gap_2.write("<h1 style='text-align: left; vertical-align: bottom;color: Green; font-size: 110%; opacity: 0.7'> 🌐 -10% ~ 5% </h1>" "Pay gap measures for every dollar paid to ethnic majority (" + eth_baseline + "), how much (less) or more goes to ethnicity minorities. For example pay gap at -10% means that on average black are paid 10% less compared to white all else equal. In US, ethnicity gap typically ranges between -10% and +5%. ", unsafe_allow_html=True)    

        num_eth_sig = df_result_eth['STAT_COUNT'].sum()
        # num_gender_sig = 0
        metric_eth_gap_3.markdown("<h1 style='text-align: left; vertical-align: bottom;color: #3498DB; font-size: 150%; opacity: 0.7'>Observation</h1>", unsafe_allow_html=True)
        if num_eth_sig>0:
            metric_eth_gap_3.markdown("<h1 style='text-align: left; vertical-align: bottom;color: Orange; font-size: 110%; opacity: 0.7'> ⚠️ Legal Risk - High </h1>", unsafe_allow_html=True)
            metric_eth_gap_3.write('There is/are '+ str(num_eth_sig) + ' statistically significant ethnicity pay gap shown with ' + '<font color=Orange> **Red** </font>' +' bar. A significant ethnicity pay gap means - we are more than 95% certain that the gap exists after incorporates all of the legitimate determinants of pay (such as differences of skill, effort, and responsibility). From a legal perspective:',unsafe_allow_html=True) 
            metric_eth_gap_3.write('* Statistically significant gap - Strong evidence of ethnicity pay discrimination' +'\n'+'* Non statistically significant gap - No evidence of ethnicity pay discrimination, the gap is likely due to random chance',unsafe_allow_html=True)
            metric_eth_gap_3.write('You may consider reducing the pay gap to a statistically insignificant level to reduce legal risk.')
        else:
            metric_eth_gap_3.markdown("<h1 style='text-align: left; vertical-align: bottom;color: Green; font-size: 110%; opacity: 0.7'> ✔️ Legal Risk - Low </h1>", unsafe_allow_html=True)
            metric_eth_gap_3.write('There is no negative statistically significant ethnicity pay gap shown in the chart. A significant ethnicity pay gap means - we are more than 95% certain that the gap exists after incorporates all of the legitimate determinants of pay (such as differences of skill, effort, and responsibility). From a legal perspective:',unsafe_allow_html=True) 
            metric_eth_gap_3.write('* Statistically significant gap - Strong evidence of ethnicity pay discrimination' +'\n'+'* Non statistically significant gap - No evidence of gender pay discrimination, the gap is likely due to random chance',unsafe_allow_html=True)
            metric_eth_gap_3.write('As a precaution, you can routinely repeat this analysis to monitor the pay gap. An alternative is to consider completely closing the pay gap to zero.')        

        # Remediation Scenarios
        main_page.markdown("""---""")
        # reme_col1, reme_col2 = main_page.columns((1, 1))
        # result_pvalue = [female_pvalue,seek_resulting_pvalues_pv,seek_resulting_pvalues_gap]

        df_reme = pd.DataFrame({'Scenario': scenario, 'How do I do this?': action, 'What is my budget?': budget, 'What is the gap after adjustment?': net_gap})

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
        styler = df_reme.style.hide_index().set_table_styles([cell_hover, index_names, headers], overwrite=False).set_properties(**{
    'white-space': 'pre-wrap'})

        main_page.markdown("<h1 style='text-align: left; vertical-align: bottom;color: #3498DB; font-size: 150%; opacity: 0.7'>Remediation Scenarios</h1>", unsafe_allow_html=True)

        # Download individual employee recommendation
        if reme_download_flag == 1:
            output_reme = BytesIO()
            writer_reme = pd.ExcelWriter(output_reme, engine='xlsxwriter')
            df_reme_ind.to_excel(writer_reme, index=False, sheet_name='Sheet1')
            workbook_reme = writer_reme.book
            worksheet_reme = writer_reme.sheets['Sheet1']

            for column in df_reme_ind:
                column_width = max(df_reme_ind[column].astype(str).map(len).max(), len(column))+3
                col_idx = df_reme_ind.columns.get_loc(column)
                writer_reme.sheets['Sheet1'].set_column(col_idx, col_idx, column_width)
            cell_format = workbook_reme.add_format()

            cell_format.set_pattern(1)  # This is optional when using a solid fill.
            cell_format.set_bg_color('yellow')
            worksheet_reme.write('A1', 'EEID',cell_format)

            writer_reme.save()
            processed_reme = output_reme.getvalue()

            # main_page.download_button(label='💰 Download Salary Adjustment',data = processed_reme,file_name= 'Salary Adjustment.xlsx')
            reme_col1, reme_col2 = main_page.columns((1, 4))
            reme_col1.markdown("🖱️ 'Save link as...'")
            reme_col2.markdown(get_excel_file_downloader_html(processed_reme, 'Salary Adjustment.xlsx'), unsafe_allow_html=True)

        main_page.write(styler.to_html(), unsafe_allow_html=True)
        st.stop()
