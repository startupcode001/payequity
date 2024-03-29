import streamlit as st
import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

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

import locale

# Helper Functions Starts here #

def rename_column(df):
    df.columns = [c.strip().upper().replace(' ', '_') for c in df.columns]
    df.columns = [c.strip().upper().replace('/', '_') for c in df.columns]
    df.columns = [c.strip().upper().replace('-', '_') for c in df.columns]
    df.columns = [c.strip().upper().replace('(', '') for c in df.columns]
    df.columns = [c.strip().upper().replace(')', '') for c in df.columns]
    df.columns = [c.strip().upper().replace('.', '') for c in df.columns]
    df.columns = [c.strip().upper().replace('___', '_') for c in df.columns]
    df.columns = [c.strip().upper().replace('__', '_') for c in df.columns]
    return df

def plot_gender_gap(coff):
    female_coff = coff
    bar_before = {'color': "grey"}
    # bar_after = {'color': "lightgreen"}
    bar_after = {'color': "Green"}
    bar_now = bar_before
    if female_coff>=-0.05:
        bar_now = bar_after
    
    fig = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        value = round(female_coff*100,1),
        mode = "gauge+number",
        number = {'suffix': "%"},
        number_font_size = 25,
        # number_font_color = '#5DADE2',
        title = {'text': ""},
        gauge = {'bar':bar_now,
                 'axis': {'range': [-20,20], 'ticksuffix':"%", 'tickmode':'linear','tick0':-20,'dtick':5 },
                 'steps' : [
                     {'range': [-20, -5], 'color': "white"},
                     {'range': [-5, 20], 'color': "lightgreen"}
                 ],
                 'threshold' : {'line': {'color': "green", 'width': 1}, 'thickness': 0.5, 'value': -5}
                }))
    fig.update_layout(autosize=False, margin=dict(l=20,r=20,b=0,t=0,pad=1), width = 300, height = 200)
                     
    return fig

def plot_full_pie(ratio,plot_type):
    if plot_type=='r2' and ratio >= 0.7:
        # color = '#5DADE2'
        color = 'Green'
    else:
        color = 'Silver'
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection':'polar'})
    data = round(ratio*100)
    data_label = str(data).strip(".0")+'%'
    startangle = 90
    x = (data * pi *2)/ 100
    left = (startangle * pi *2)/ 360 #this is to control where the bar starts
    print(left)
    plt.xticks([])
    plt.yticks([])
    ax.spines.clear()
    ax.barh(1, x, left=left, height=1.5, color=color) 
    plt.ylim(-3, 3)
    plt.text(0, -3, data_label, ha='center', va='center', fontsize=25)
    return fig

def plot_half_pie(ratio,ratio_max, plot_type):
    # data
    label = [plot_type, ""]
    val = [ratio,ratio_max-ratio,]

    # append data and assign color
    label.append("")
    val.append(sum(val))  # 50% blank
    colors = ['red', 'blue', 'green', 'white']

    # plot
    fig = plt.figure(figsize=(8,6),dpi=100)
    ax = fig.add_subplot(1,1,1)
    ax.pie(val, labels=label, colors=colors)
    ax.add_artist(plt.Circle((0, 0), 0.6, color='white'))
    fig.show()


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
        warning_feature = feature+": exclude "+str(exclude_feature_num)+" invalid records"
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
        warning_feature = feature + ": exclude this optional factor"
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

def run(data=None):
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
    df.columns = [c.strip().upper().replace(' ', '_') for c in df.columns]
    df.columns = [c.strip().upper().replace('/', '_') for c in df.columns]
 
    # 2.1 Data Cleaning ****************************
    # Data Validation
    df['VALIDATION_MESSAGE']=""
    df['VALIDATION_FLAG']=0
    # Snapshot Date
    try:
        df['SNAPSHOT_DATE'] = df['SNAPSHOT_DATE'].astype("string")
        snapshot = df['SNAPSHOT_DATE'].mode().tolist()[0]
    except:
        error_snapshot = "Invalid snapshot date, Please check submission format is mm/dd/yyyy in data template"
        error_message['SNAPSHOT_DATE'] = error_snapshot
    df['NOW'] = pd.to_datetime(snapshot)
        
    # 2.2 Clean up All features ******************
    df,warning_message = clean_req_feature(data = df,feature = "EEID",valid_feature_list=[],warning_message = warning_message, data_type="string")
    df,warning_message = clean_req_feature(data = df,feature = "GENDER",valid_feature_list=["F","M","O"],warning_message = warning_message, data_type="string")
    df,warning_message = clean_req_feature(data = df,feature = "SALARY",valid_feature_list=[],warning_message = warning_message, data_type='numeric')
    df,warning_message = clean_req_feature(data = df,feature = "JOB_LEVEL_OR_COMP_GRADE",valid_feature_list=[],warning_message = warning_message, data_type="string")
    df,warning_message = clean_req_feature(data = df,feature = "JOB_FUNCTION",valid_feature_list=[],warning_message = warning_message,data_type="string")
    df,warning_message = clean_req_feature(data = df,feature = "STATE",valid_feature_list=[],warning_message = warning_message,data_type="string")
    df,warning_message = clean_req_feature(data = df,feature = "FULL_TIME",valid_feature_list=["Y","N"],warning_message = warning_message,data_type="string")
    df,warning_message = clean_req_feature(data = df,feature = "FLSA_EXEMPT",valid_feature_list=["Y","N"],warning_message = warning_message,data_type="string")

    # Clean up optional features
    df,warning_message, exclude_col = clean_optional_feature(data = df,feature = "ETHNICITY",valid_feature_list=[],warning_message = warning_message,exclude_col = exclude_col, data_type="string")
    df,warning_message, exclude_col = clean_optional_feature(data = df,feature = "PEOPLE_MANAGER",valid_feature_list=["Y","N"],warning_message = warning_message,exclude_col = exclude_col, data_type="string")
    df,warning_message, exclude_col = clean_optional_feature(data = df,feature = "EDUCATION",valid_feature_list=[],warning_message = warning_message,exclude_col = exclude_col, data_type="string")
    df,warning_message, exclude_col = clean_optional_feature(data = df,feature = "PROMOTION",valid_feature_list=["Y","N"],warning_message = warning_message,exclude_col = exclude_col, data_type="string")
    df,warning_message, exclude_col = clean_optional_feature(data = df,feature = "PERFORMANCE",valid_feature_list=[],warning_message = warning_message,exclude_col = exclude_col, data_type="string")
    df,warning_message, exclude_col = clean_optional_feature(data = df,feature = "DATE_OF_BIRTH",valid_feature_list=[],warning_message = warning_message,exclude_col = exclude_col, data_type="datetime")
    df,warning_message, exclude_col = clean_optional_feature(data = df,feature = "DATE_OF_HIRE",valid_feature_list=[],warning_message = warning_message,exclude_col = exclude_col, data_type="datetime")
    
    # Record Message
    df_org = df.copy()
    df_org.to_excel('Data\data_validate.xlsx')
    before_clean_record = df_org.shape[0]
    
    df = df_org[df_org['VALIDATION_MESSAGE']==""]
    after_clean_record=df.shape[0]

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
    
    for c in numeric_columns:
        df.loc[df[c]=='na', c] = np.nan
        df[c] = pd.to_numeric(df[c])
    
    # df.to_excel('edu.xlsx')
    
    # %% Convert string columns to categories
    category_columns = [x for x in col_list if x not in numeric_columns]
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
    
    # 3 Modeling - Linear Regression ****************************
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

    df_result = results.summary2().tables[1]
    df_result.reset_index(level=0, inplace=True)
    df_result = df_result.rename(columns={"index":"CONTENT"})

    r2 = results.rsquared
    r2_adj = results.rsquared_adj

    female_coff = df_result.loc[df_result['CONTENT']=="GENDER[T.F]"]['Coef.'].tolist()[0]
    female_pvalue = df_result.loc[df_result['CONTENT']=="GENDER[T.F]"]['P>|t|'].tolist()[0]
    
    # Gender Net Gap - Prediciton
    y_predict, x_predict = dmatrices(f_predict, df, return_type='dataframe')
    model_predict = sm.OLS(y_predict, x_predict)
    results_predict = model_predict.fit()

    y_pred = results_predict.predict(x_predict)
    std, lower, upper = wls_prediction_std(results_predict)
    
    # Save budget file for prediction
    X_full = x_dis    
    budget_df = pd.DataFrame({'EEID':df['EEID'], 'original': df['LOG_SALARY'], 'GENDER': df['GENDER'],'predicted':y_pred,'pred_lower': lower, 'pred_upper': upper, 'pred_stderr': std})
#     X_full.to_excel('xfull.xlsx')
#     df.to_excel('check_final.xlsx')
#     budget_df.to_excel('check_final_budget.xlsx')
    # r2 = 0.91
    # Graphs
    fig_r2_gender_gap = plot_full_pie(r2,'r2')
    fig_raw_gender_gap = plot_gender_gap(female_coff)
    fig_net_gender_gap = plot_gender_gap(female_coff)
    
    # Statistics for output
    hc_female = df[(df['GENDER']=='F') | (df['GENDER']=='FEMALE')].shape[0]
    
    # print(message.loc[['OVERVIEW']]['Message'])
    
    return df, df_org, message, exclude_col, r2_raw, female_coff_raw, female_pvalue_raw, r2, female_coff, female_pvalue, before_clean_record, after_clean_record,hc_female,fig_r2_gender_gap,fig_raw_gender_gap,fig_net_gender_gap,X_full,budget_df


def reme(df,budget_df,X_full,factor, project_group_feature, protect_group_class):
    budget_df['adj_lower'] = budget_df['predicted'] - factor * budget_df['pred_stderr']

    # Adjust protect group pay only, others set to original pay
    budget_df['adj_salary'] = budget_df['original']
    budget_df.loc[(budget_df[project_group_feature] == protect_group_class) & (budget_df['original'] < budget_df['adj_lower']),'adj_salary'] = budget_df['adj_lower']
    
    # Recalculate pay gap and p value with adjusted salary
    model = sm.OLS(budget_df['adj_salary'], X_full)
    results = model.fit()

    budget = np.sum(np.exp(budget_df['adj_salary']) - np.exp(budget_df['original']))
    budget_df['S_Salary'] = np.exp(budget_df['original'])
    budget_df['S_Budget'] = np.exp(budget_df['adj_salary'])-np.exp(budget_df['original'])
    budget_df['S_Adjusted'] = np.exp(budget_df['adj_salary'])
    budget_df['S_AdjInd'] = 0
    budget_df.loc[budget_df['S_Budget'] >0, 'S_AdjInd']=1

    # Reporting
    current_total_salary = np.sum(budget_df['S_Salary'])
    Budget_PCT = budget_df['S_Budget']/np.exp(budget_df['original'])

    target_position = 1
    resulting_gap = results.params[target_position]
    resulting_pvalues = results.pvalues[target_position]
    adj_count = budget_df['S_AdjInd'].sum()
    adj_average = Budget_PCT[Budget_PCT>0].mean()
    adj_max = Budget_PCT[Budget_PCT>0].max()
    adj_budget_pct = budget/current_total_salary
    
    # print(results.summary())
    # print(resulting_gap)
    # print(resulting_pvalues)
    # print(adj_count)
    # print(adj_budget_pct)
    # budget_df.to_excel('check_final_budget.xlsx')

    return budget_df, budget, resulting_gap, resulting_pvalues, adj_count, adj_budget_pct

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
def reme_gap_seek(df,budget_df,X_full, project_group_feature, protect_group_class, seek_goal, current_pvalue, current_gap, search_step = -0.001):
    factor_range = np.arange(2, -2,search_step)
    threshold = 0.0005
    
    seek_budget_df = np.nan
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
    
    return seek_budget_df,seek_budget,seek_resulting_gap,seek_resulting_pvalues,seek_adj_count, seek_adj_budget_pct,seek_pass,seek_success
    
@st.experimental_memo(show_spinner=False)
# Run Goal Seek for insignificant gap and 0 gap
def reme_pvalue_seek(df,budget_df,X_full, project_group_feature, protect_group_class, seek_goal, current_pvalue, current_gap, search_step= -0.005):
    
    factor_range = np.arange(2, -2,search_step)
    threshold = 0.0005
    
    seek_budget_df = np.nan
    seek_budget = np.nan
    seek_resulting_gap  = np.nan
    seek_resulting_pvalues =  np.nan
    seek_adj_count = np.nan
    seek_adj_budget_pct = np.nan
    seek_pass = False
    seek_success = False
    
    if current_pvalue>=0.05:
        print('Current P value already greater than 5%: '+str(current_pvalue))
        seek_pass = False
        seek_resulting_gap = current_gap
    else:
        seek_pass = True
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
    
    return seek_budget_df,seek_budget,seek_resulting_gap,seek_resulting_pvalues,seek_adj_count, seek_adj_budget_pct,seek_pass,seek_success

def analysis(df_submit, run_demo, demo_path, main_page, main_page_info):
    # Process df (not demo datafile)    
    with st.spinner('Running analysis, Please wait for it...'):
        m_info = main_page_info.success('Reading Data')
        if run_demo == True:
            # Demo Run
            df = pd.read_excel(demo_path,sheet_name="Submission")
        else:
            df = pd.read_excel(df_submit,sheet_name="Submission")
            
        # Run discovery model:
        m_info = main_page_info.success('Running Gap Analysis')
        df, df_org, message, exclude_col, r2_raw, female_coff_raw, female_pvalue_raw, r2, female_coff, female_pvalue, before_clean_record, after_clean_record,hc_female,fig_r2_gender_gap,fig_raw_gender_gap,fig_net_gender_gap,X_full,budget_df = run(df)
        
        print('pvalue'+str(female_pvalue))
        
        # Run Reme Pvalue = 7%
        m_info = main_page_info.success('Running Remediation Scenario A: Mitigate Legal Risk')
        seek_budget_df_pv,seek_budget_pv,seek_resulting_gap_pv,seek_resulting_pvalues_pv,seek_adj_count_pv, seek_adj_budget_pct_pv,seek_pass_pv, seek_success_pv = reme_pvalue_seek(df,budget_df,X_full, project_group_feature='GENDER', protect_group_class='F', seek_goal=0.07, current_gap = female_coff, current_pvalue = female_pvalue, search_step = -0.005)
        if seek_success_pv == False:
            seek_budget_df_pv,seek_budget_pv,seek_resulting_gap_pv,seek_resulting_pvalues_pv,seek_adj_count_pv, seek_adj_budget_pct_pv,seek_pass_pv, seek_success_pv = reme_pvalue_seek(df,budget_df,X_full, project_group_feature='GENDER', protect_group_class='F', seek_goal=0.07, current_gap = female_coff, current_pvalue = female_pvalue, search_step = -0.001)
        
        print('pvalue'+str(seek_resulting_pvalues_pv))
        
        # Run Reme Zero Gap
        m_info = main_page_info.success('Running Remediation Scenario B: Close Gender Gap')
        seek_budget_df_gap,seek_budget_gap,seek_resulting_gap_gap,seek_resulting_pvalues_gap,seek_adj_count_gap, seek_adj_budget_pct_gap,seek_pass_gap, seek_success_gap = reme_gap_seek(df,budget_df,X_full, project_group_feature='GENDER', protect_group_class='F', seek_goal=0, current_gap = female_coff, current_pvalue = female_pvalue, search_step = -0.005)
        if seek_success_gap == False:
            seek_budget_df_gap,seek_budget_gap,seek_resulting_gap_gap,seek_resulting_pvalues_gap,seek_adj_count_gap, seek_adj_budget_pct_gap,seek_pass_gap, seek_success_gap = reme_gap_seek(df,budget_df,X_full, project_group_feature='GENDER', protect_group_class='F', seek_goal=0, current_gap = female_coff, current_pvalue = female_pvalue, search_step = -0.001)

        print('pvalue'+str(seek_resulting_pvalues_gap))    
        
        # Run data validation
        m_info = main_page_info.success('Output Data Validation')
        demo_validation = convert_df(df_org)
                
        # Display run is successful message    
        m_info = main_page_info.success('View Result: '+message.loc[['OVERVIEW']][0])

        main_page.markdown("""---""")
        m_col1_but_col1, m_col1_but_col2, m_col1_but_col3, m_col1_but_col4 = main_page.columns((2, 2, 2, 1))

        # Display headcount, Successful Run, Female Percent, download validation file
        m_col1_but_col1.metric('💬 Submission Record',before_clean_record)
        m_col1_but_col2.metric('🏆 Successful Run',after_clean_record)
        m_col1_but_col3.metric('👩 Female Headcount %',round(hc_female/after_clean_record,2)*100)
        m_col1_but_col4.download_button('📥 Download exclusions', data=demo_validation, file_name='Data Validation.csv',mime='text/csv')
        
        # r2= 0.9
        
        # Show R2
        main_page.markdown("""---""")
        metric_R2_1, metric_R2_2, metric_R2_3 = main_page.columns((1, 1.7, 1.7))            
        metric_R2_1.markdown("<h1 style='text-align: left; vertical-align: bottom; font-size: 150%; color: #3498DB; opacity: 0.7'>Robustness</h1>", unsafe_allow_html=True)
        metric_R2_1.pyplot(fig_r2_gender_gap, use_container_width=True)

        metric_R2_2.markdown("<h1 style='text-align: left; vertical-align: bottom;color: #3498DB; font-size: 150%; opacity: 0.7'>Benchmark</h1>", unsafe_allow_html=True)
        metric_R2_2.markdown("<h1 style='text-align: left; vertical-align: bottom;color: Green; font-size: 110%; opacity: 0.7'> 🌐 70% ~ 100%  </h1>" "  \n"  "Model Robutness measures how well the standard model explain pay decisions. For example 80% means the standard model explains 80 percent of the pay variation among employees.", unsafe_allow_html=True)

        metric_R2_3.markdown("<h1 style='text-align: left; vertical-align: bottom;color: #3498DB; font-size: 150%; opacity: 0.7'>Observation</h1>", unsafe_allow_html=True)
        if r2>=0.7:
            metric_R2_3.markdown("<h1 style='text-align: left; vertical-align: bottom;color: Green; font-size: 110%; opacity: 0.7'> ✔️ Align with market  </h1>", unsafe_allow_html=True)
            metric_R2_3.markdown("The model is **robust** to explain pay variation. Let's see the pay gap results.", unsafe_allow_html=True)
        else:
            metric_R2_3.markdown("<h1 style='text-align: left; vertical-align: bottom;color: Orange; font-size: 110%; opacity: 0.7'> ⚠️ Below market  </h1>" , unsafe_allow_html=True)
            metric_R2_3.markdown("Our standard model is **NOT Sufficient** to make pay gap conclusions. We can improve robustness by including additional pay factors, such as high potential, cost center, skills, etc. Please contact us for a free consultation.", unsafe_allow_html=True)
            st.stop()
        # metric_R2_3.markdown("<h1 style='text-align: center; vertical-align: bottom; color: Black; background-color: #3498DB; opacity: 0.7; border-style: dotted'>Observation</h1>", unsafe_allow_html=True)
        
        # Show Net Gap
        main_page.markdown("""---""")
        metric_net_gap_1, metric_net_gap_2, metric_net_gap_3 = main_page.columns((1, 1.6, 1.6))            
        metric_net_gap_1.markdown("<h1 style='text-align: left; vertical-align: bottom; font-size: 150%; color: #3498DB; opacity: 0.7'>Gender Net Gap</h1>", unsafe_allow_html=True)
        metric_net_gap_1.plotly_chart(fig_net_gender_gap, use_container_width=True)

        metric_net_gap_2.markdown("<h1 style='text-align: left; vertical-align: bottom;color: #3498DB; font-size: 150%; opacity: 0.7'>Benchmark</h1>", unsafe_allow_html=True)
        metric_net_gap_2.write("<h1 style='text-align: left; vertical-align: bottom;color: Green; font-size: 110%; opacity: 0.7'> 🌐 > -5% </h1>" "For every 1 dollar paid to male employees, how much (lesser)/more is paid to female employees. For example -3% means on average female employees is paid 3% LOWER than male employees all else equal. Typically the net pay gap in US is between -5% and +1%", unsafe_allow_html=True)

        metric_net_gap_3.markdown("<h1 style='text-align: left; vertical-align: bottom;color: #3498DB; font-size: 150%; opacity: 0.7'>Observation</h1>", unsafe_allow_html=True)
        
#         female_pvalue = 0.04
#         female_coff = 0.02
        
        print(female_coff)
        print(female_pvalue)
        if female_pvalue>0.05:
            metric_net_gap_3.markdown("<h1 style='text-align: left; vertical-align: bottom;color: Green; font-size: 110%; opacity: 0.7'> ✔️ Legal Risk - Low </h1>", unsafe_allow_html=True)
            metric_net_gap_3.markdown("Congratulation! gender gap is **NOT statistically significant**. Your **legal risk is mimiumum** and defensible on sound statistical grounds", unsafe_allow_html=True)
            if female_coff<-0.05:
                metric_net_gap_3.markdown("<h1 style='text-align: left; vertical-align: bottom;color: Orange; font-size: 110%; opacity: 0.7'> ⚠️ Pay Gap - Below market  </h1>", unsafe_allow_html=True)
                metric_net_gap_3.markdown("Your pay gap is below market, we suggest to periodically rerun to confirm legal risk. Alternatively you may consider to **fully close pay gap** at 0%", unsafe_allow_html=True)    
            elif female_coff>=-0.05 and female_coff<0:
                metric_net_gap_3.markdown("<h1 style='text-align: left; vertical-align: bottom;color: Green; font-size: 110%; opacity: 0.7'> ✔️ Pay Gap - Align with market  </h1>", unsafe_allow_html=True)
                metric_net_gap_3.markdown("Your pay gap is align with market. To be the market leader, you may consider to **fully close pay gap** at 0%", unsafe_allow_html=True)
            else:
                metric_net_gap_3.markdown("<h1 style='text-align: left; vertical-align: bottom;color: Green; font-size: 110%; opacity: 0.7'> ✔️ Pay Gap - Market leader!  </h1>", unsafe_allow_html=True)
                metric_net_gap_3.markdown("Congratulation! Your female employees on average **earn more** than male employees. Only 1% companies achieved your great status!", unsafe_allow_html=True)
        else:
            if female_coff>0:
                metric_net_gap_3.markdown("<h1 style='text-align: left; vertical-align: bottom;color: Green; font-size: 110%; opacity: 0.7'> ✔️ Legal Risk - Low </h1>", unsafe_allow_html=True)
                metric_net_gap_3.markdown("Congratulation! Your female employees on average **earn more** than male employees. Only 1% companies achieved your great status!", unsafe_allow_html=True)
            else:
                metric_net_gap_3.markdown("<h1 style='text-align: left; vertical-align: bottom;color: Orange; font-size: 110%; opacity: 0.7'> ⚠️ Legal Risk - High </h1>", unsafe_allow_html=True)
                metric_net_gap_3.markdown('Gender gap is **statistically significant**, you are vulnerable to gender equality litigation.', unsafe_allow_html=True)
                metric_net_gap_3.markdown("<h1 style='text-align: left; vertical-align: bottom;color: Orange; font-size: 110%; opacity: 0.7'> ⚠️ Pay Gap - Below market  </h1>", unsafe_allow_html=True)
                metric_net_gap_3.markdown("To lower legal risk, you should **reduce pay gap** to statistically insignificant level", unsafe_allow_html=True)

        main_page.markdown("""---""")
        overview_1, overview_2 = main_page.columns((1, 3))
        overview_1.image('Picture/overview.jpg',use_column_width='auto')
        if female_pvalue>0.05:
            overview_2.markdown("<h1 style='text-align: left; vertical-align: bottom;color: Green; font-size: 110%; opacity: 0.7'>  Congratulation!  </h1>", unsafe_allow_html=True)
            if female_coff<-0.05:
                overview_2.markdown('Your pay gap is at <font color=Green> **low** </font> legal risk. However, your pay gap is <font color=Green> **larger** </font> than market. Larger negative pay gap usually leads to statistically significant status which increase legal risk. As a preventive action, you may **periodically rerun** this analysis to monitor pay gap. Alternative, you may consider to **full close** pay gap - See Scenario B below', unsafe_allow_html=True)
            elif female_coff>=-0.05 and female_coff<0:
                overview_2.markdown('Your pay gap is at <font color=Green> **low** </font> legal risk. You are also **aligned** with market! We recommend to **monitor pay gap periodically** - for instance before and after merit increase, M&A, organization restructure, and major job releveling. Also you may consider to **full close** pay gap - See Scenario B below', unsafe_allow_html=True)
            else:
                overview_2.markdown('Your pay gap is at <font color=Green> **low** </font> legal risk. You are the <font color=Green> **market leader** </font> in gender pay equaity (only 1% of companies achieve female employee pays more than male employee all else equal. We recommend to **monitor pay gap periodically** - for instance before and after merit increase, M&A, organization restructure, and major job releveling.', unsafe_allow_html=True)
        else:
            if female_coff>0:
                overview_2.markdown("<h1 style='text-align: left; vertical-align: bottom;color: Green; font-size: 110%; opacity: 0.7'>  Congratulation!  </h1>", unsafe_allow_html=True)
                overview_2.markdown('Your pay gap is at <font color=Green> **low** </font> legal risk. You are the <font color=Green> **market leader** </font> in gender pay equaity (only 1% of companies achieve female employee pays more than male employee all else equal. We recommend to **monitor pay gap periodically** - for instance before and after merit increase, M&A, organization restructure, and major job releveling.', unsafe_allow_html=True)
            else:
                overview_2.markdown("<h1 style='text-align: left; vertical-align: bottom;color: Orange; font-size: 110%; opacity: 0.7'>  Action Needed!  </h1>", unsafe_allow_html=True)
                overview_2.markdown('Your pay gap is at <font color=Orange> **high** </font> legal risk. You should consider to **reduce pay gap** to statistically insignificant level - See **Secnario A** below. Alternatively you may also consider to **full close** pay gap at a higher cost - See **Scenario B** below', unsafe_allow_html=True)
        main_page.markdown("""---""")

        message_budget_pv = np.nan
        if seek_pass_pv == False:
            message_budget_pv = '0 - current gap is already statistically insignificant'
        elif (seek_pass_pv == True) and (seek_success_pv == False):
            message_budget_pv = 'No result is found, please contact consultant for more detail'
        else:
            message_budget_pv = str(locale.format("%d", round(seek_budget_pv/1000,0), grouping=True))+'K'+'\n'+'('+str(round(seek_adj_budget_pct_pv*100,0))+'% of Pay)'

        message_budget_gap = np.nan
        if seek_pass_gap == False:
            message_budget_gap = '0 - current gap is already greater than zero, no futher adjustment is needed'
        elif (seek_pass_gap == True) and (seek_success_gap == False):
            message_budget_gap = 'No result is found, please contact consultant for more detail'
        else:
            message_budget_gap = str(locale.format("%d", round(seek_budget_gap/1000,0), grouping=True))+'K'+'\n'+'('+str(round(seek_adj_budget_pct_gap*100,2))+'% of Pay)'

        scenario = ['Current','A','B']
        action = ['🏁 No change','✔️ Mitigate legal risk \n'+'- Reduce gender pay gap to statistically insignificant level','✔️ Mitigate legal risk \n'+'✔️✔️ Completely close gender pay gap \n']
        budget = ['0',message_budget_pv,message_budget_gap]
        net_gap = [female_coff,seek_resulting_gap_pv,seek_resulting_gap_gap]
        net_gap = [f'{i*100:.1f}%' for i in net_gap]

        # result_pvalue = [female_pvalue,seek_resulting_pvalues_pv,seek_resulting_pvalues_gap]

        df_reme = pd.DataFrame({'Scenario': scenario, 'What is the action?': action, 'How much does it cost?': budget, 'What is the gap now?': net_gap})

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
        main_page.write(styler.to_html(), unsafe_allow_html=True)