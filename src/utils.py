# Data Analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Extracting Geological information
from geopy.geocoders import Nominatim
from geopy.distance import great_circle as GRC


# Model evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score,roc_auc_score, roc_curve, precision_score, recall_score, plot_confusion_matrix
from scikitplot.metrics import plot_roc_curve as auc_roc
np.random.seed(10)

def datetime_split(date_str):
    date_time = date_str.split()
    date_time = date_time[0].split('-') + date_time[1].split(':')
    date_time = [int(x) for x in date_time]
    time_s = [date_time[3]*3600 + date_time[4]*60 + date_time[5]]
    return date_time[:3] + time_s

# Optimized code where I consider month and day of 
def min_diff(date_i, date_o, seconds_in_day = 24*3600):
    schedule_date_sec = date_i[2]*seconds_in_day
    
    if date_i[1] == date_o[1]:
        if date_i[2] == date_o[2]:
            time_diff = date_o[3] - date_i[3]
        elif date_i[2] < date_o[2]:
            time_diff = (schedule_date_sec+ date_o[3]) - ((schedule_date_sec - seconds_in_day) + date_i[3])
        else:
            time_diff = (schedule_date_sec - (seconds_in_day-date_o[3])) - (schedule_date_sec + date_i[3])
    else:
        if date_i[2] > date_o[2]:
            time_diff = (schedule_date_sec+ date_o[3]) - ((schedule_date_sec - seconds_in_day) + date_i[3])
        else:
            time_diff = (schedule_date_sec - (seconds_in_day-date_o[3])) - (schedule_date_sec + date_i[3])
    
    return time_diff/60

def high_season(month, day):
    month_season = {'high': [1, 2], 
                    'low': [4, 5, 6, 8, 10, 11], 
                    'mid': {3: list(range(1,4)), 7: list(range(15,32)), 9: list(range(11,31)), 12: list(range(15,32))}}

    # Checks high and low months
    if month in month_season['high']:
        return 1
    elif month in month_season['low']:
        return 0

    # Checks if day of Month with temporary high/low seasons
    if day in month_season['mid'][month]:
        return 1
    else:
        return 0

def delay_15(time_diff):
    return 1 if time_diff>15 else 0

def period_day(time, sec_in_hour = 3600):
    period_of_day = {'morning': [5*sec_in_hour, 12*sec_in_hour],
                    'afternoon': [12*sec_in_hour, 19*sec_in_hour],
                    'night': [19*sec_in_hour, 5*sec_in_hour]}
    
    if time >= period_of_day['morning'][0] and time < period_of_day['morning'][1]:
        return 'morning'
    elif time >= period_of_day['afternoon'][0] and time < period_of_day['afternoon'][1]:
        return 'afternoon'
    elif time >= period_of_day['night'][0] or time < period_of_day['night'][1]:
        return 'night'

#Classification Summary Function
def classification_summary(pred,pred_prob,y_test, model):
  print('{}{}\033[1m Evaluating {} \033[0m{}{}\n'.format('<'*3,'-'*35,model, '-'*35,'>'*3))
  print('Accuracy = {}%'.format(round(accuracy_score(y_test, pred),3)*100))
  print('F1 Score = {}%'.format(round(f1_score(y_test, pred, average='weighted'),3)*100))
  print('Precision Score = {}%'.format(round(precision_score(y_test, pred, average='weighted'),3)*100))
  print('Recall Score = {}%'.format(round(recall_score(y_test, pred, average='weighted'),3)*100))
  print('AUC-ROC score = {}%'.format(round(roc_auc_score(y_test, pred_prob[:,1], multi_class='ovr'),3)*100))
  
  print('\n \033[1mConfusiton Matrix:\033[0m\n',confusion_matrix(y_test, pred))
  print('\n\033[1mClassification Report:\033[0m\n',classification_report(y_test, pred))
  