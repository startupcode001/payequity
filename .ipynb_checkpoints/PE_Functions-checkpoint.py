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
    bar_before = {'color': "lightgrey"}
    bar_after = {'color': "green"}
    bar_now = bar_before
    if female_coff>=0:
        bar_now = bar_after
    
    fig = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        value = 1+round(female_coff,2),
        mode = "gauge+number+delta",
        title = {'text': "Women earn {} dollar for every 1 dollar earn by Men".format(1+round(female_coff,2))},
        delta = {'reference': 1},
        gauge = {'bar':bar_now,
                 'axis': {'range': [0.8,1.2]},
                 'steps' : [
                     {'range': [0.95, 1.2], 'color': "lightgreen"},
                     {'range': [0.8, 0.95], 'color': "white"}
                 ],
                 'threshold' : {'line': {'color': "blue", 'width': 1}, 'thickness': 0.5, 'value': 1}
                }))
    return fig

def clean_req_feature(data, feature, valid_feature_list, warning_message, data_type="string", 
                      val_col = 'VALIDATION_MESSAGE',val_flag_col = 'VALIDATION_FLAG'):
    da = data.copy()
    # Check exclusion and remove invalid data
    if data_type == "numeric":
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
    # Use Starting Salary or not, Y/N
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
    df = df.set_index('EEID', drop=True)
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
    
    df.to_excel('edu.xlsx')
    
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
    add_exclude_col = ['SALARY','SNAPSHOT_DATE', 'VALIDATION_MESSAGE', 'VALIDATION_FLAG', 'NOW','DATE_OF_BIRTH','DATE_OF_HIRE']
    add_exclude_col_predict = add_exclude_col+['GENDER','ETHNICITY']
    exclude_col = exclude_col+add_exclude_col
    exclude_col_predict = exclude_col+add_exclude_col_predict

    model_col = [x for x in col_list if x not in exclude_col]
    model_col_predict = [x for x in col_list if x not in exclude_col_predict]
    
    # Factors
    f_raw = 'LOG_SALARY ~ GENDER'
    f_discover = model_col[-1] + ' ~ ' + ' + '.join(map(str, model_col[0:len(model_col)-1]))
    f_predict= model_col[-1] + ' ~ ' + ' + '.join(map(str, model_col_predict[0:len(model_col_predict)-1]))
    
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

    # Graphs
    plot_gender = plot_gender_gap(female_coff)
    
    return df, df_org, message, exclude_col, r2_raw, female_coff_raw, female_pvalue_raw, r2, female_coff, female_pvalue, plot_gender