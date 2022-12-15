# Data Analysis
import pandas as pd
# Pre processing
from sklearn import preprocessing
# Data prep
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from utils import *

class DataProcessor:
    def __init__(self):
        self.le = preprocessing.LabelEncoder()
        self.oversample = SMOTE(k_neighbors=5)

    def load_csv(self, data_path):
        return pd.read_csv(data_path)

    def generate_synthetic_features(self, df, features):
        syn_df = df[features]
        syn_df = syn_df.applymap(datetime_split)
        syn_df['high_season'] = syn_df.apply(lambda x: high_season(month=x['Fecha-I'][1], day=x['Fecha-I'][2]), axis=1)
        syn_df['min_diff'] = syn_df.apply(lambda x: min_diff(date_i=x['Fecha-I'], date_o=x['Fecha-O']), axis=1)
        syn_df['delay_15'] = syn_df.apply(lambda x: delay_15(x['min_diff']), axis=1)
        syn_df['period_day'] = syn_df.apply(lambda x: period_day(x['Fecha-I'][3]), axis=1)
        syn_df['sched_hour'] = syn_df.apply(lambda x: x['Fecha-I'][3]/3600, axis=1)
        syn_df['op_hour'] = syn_df.apply(lambda x: x['Fecha-O'][3]/3600, axis=1)
        return syn_df[['min_diff', 'high_season', 'delay_15', 'period_day', 'sched_hour', 'op_hour']]

    def merge_dataframe(self, df_lst):
        return pd.concat(df_lst, axis=1)

    def extract_and_prepare(self, df, target_features):
        target_df = df[target_features]
        target_df['Vlo-I'] = target_df['Vlo-I'].apply(lambda x: pd.to_numeric(x, errors='coerce'))
        return target_df
    
    def encode_data(self, df):
        return df.apply(self.le.fit_transform)

    def split_and_oversample(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        oversample = SMOTE(k_neighbors=5)
        X_smote, y_smote = oversample.fit_resample(X_train, y_train)
        X_train, y_train = X_smote, y_smote
        return X_train, X_test, y_train, y_test

if __name__ == '__main__':
   
    target_data = ['sched_hour', 'op_hour', 'Ori-I', 'Vlo-I', 'Des-I', 'Emp-I', 'DIANOM', 'TIPOVUELO', 'OPERA', 'SIGLAORI', 'SIGLADES', 'DIA', 'MES','period_day', 'high_season', 'delay_15']
    data = DataProcessor()

    df = data.load_csv('../data/dataset_SCL.csv')
    syn_feat = data.generate_synthetic_features(df, ['Fecha-I', 'Fecha-O'])

    syn_feat[['min_diff', 'high_season', 'delay_15', 'period_day']].to_csv('../data/synthetic_features_v2.csv', index=False)
    syn_feat[['sched_hour', 'op_hour']].to_csv('../data/additional_features_v2.csv', index=False)