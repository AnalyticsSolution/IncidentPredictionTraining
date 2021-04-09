# Loading the libraries
 

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
# from sklearn.externals import joblib
# import sklearn.external.joblib as extjoblib
import configparser
import calendar
import datetime
import urllib
import random
import joblib

# Reading configuration file

config = configparser.RawConfigParser()

try:
    config.read('Configuration.txt')
except Exception as e:
    print(str(e))
try:
    
    model_path = config.get('Paths', 'model_path')
    metric_path = config.get('Paths', 'metric_path')
    le_path = config.get('Paths', 'le_path')
    input_path = config.get('Paths', 'input_path')
    output = config.get('Paths', 'output_path')
    date = config.get('Date', 'prediction_date')
    pre_range = config.get('Date', 'prediction_range')

except Exception as e:
    print('Could not read configuration file. {}'.format(str(e)))

# conecting to database server retriving data

print("Connecting to data")


# INC_df = pd.read_sql_query(query, con)
INC_df = pd.read_csv(r'../Input/INC_df_2.csv')

# Function for prediction


def prediction(df, j='User Created'):

    # Label encoding
    
    df1 = df.iloc[:, 1:]
    for i in df1.columns:
        if (df1[i].dtype == 'object'):
            let = joblib.load(le_path + i + j + '.pkl')
            df1[i] = let.transform(df1[i])

    x = df1

    # Loading models

    DTR = joblib.load(model_path+'1_DTR_'+ j +'.pkl')
    KNNR = joblib.load(model_path+'1_KNNR_'+ j +'.pkl')
    RFR = joblib.load(model_path+'1_RFR_'+ j +'.pkl')

    df['discover source'] = j

    # Predicting data

    DTR_pred = np.array(DTR.predict(x))
    KNNR_pred = np.array(KNNR.predict(x))
    RFR_pred = np.array(RFR.predict(x))


    df['Predicted_incidents'] = ((2*DTR_pred) + KNNR_pred + RFR_pred)/4
    df['Predicted_incidents'] = df['Predicted_incidents'].astype('int64')

    return df

# Creating date list for prediction

current_dt = datetime.datetime.strptime(date, "%d-%m-%Y")
date_list = [current_dt + datetime.timedelta(days=x) for x in range(1,1+int(pre_range))]
date_list = [x.date() for x in date_list]


# Fetching 5 combination randomly from past incidents

date_list_2 = [current_dt - datetime.timedelta(days=x) for x in range(1,101)]
str_dates = [date_obj.strftime('%Y-%m-%d') for date_obj in date_list_2]

custom_date = list(INC_df[INC_df['date'].isin(str_dates)]['date'])
random_5_dt = random.choices(custom_date,k=5)

# Creating prediction data

INC_df = INC_df[INC_df['date'].isin(random_5_dt)]

# removing outliers

df_1 = INC_df[(INC_df['No_of_incidents'] < 30)]

# Updating date, month and day on picked combinations

df_3 = pd.DataFrame()
for i,j in zip(date_list,random_5_dt):
    df_2 = df_1[df_1['date']==j]
    df_2['date'] = i
    df_2['month'] = calendar.month_abbr[i.month]
    df_2['day'] = calendar.day_name[i.weekday()]
    df_3 = pd.concat([df_3,df_2])


 
prediction_df = df_3[['date', 'month', 'day', 'BH', 'province', 'customerseverity', 'devicetype_en','service name_en']]

# Predicting incidents

print('Predicting data...')

output = prediction(prediction_df)

# Exporting output to DB

output = output[['date', 'month', 'day', 'BH', 'province', 'customerseverity', 'devicetype_en',
                 'service name_en','discover source', 'Predicted_incidents']]


if os.path.isfile(r'../Output/prediction_output.csv'):
    output.to_csv(r'../Output/prediction_output.csv', index=False,mode='a',header=False)
else:
    output.to_csv(r'../Output/prediction_output.csv', index=False,header=True)

print('Prediction done.')
