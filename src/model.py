from sklearn.ensemble import RandomForestClassifier

from utils import classification_summary
from data_processing import DataProcessor

import pickle

if __name__ == '__main__':
    
    target_data = ['sched_hour', 'op_hour', 'Ori-I', 'Vlo-I', 'Des-I', 'Emp-I', 'DIANOM', 'TIPOVUELO', 'OPERA', 'SIGLAORI', 'SIGLADES', 'DIA', 'MES','period_day', 'high_season', 'delay_15']
    data = DataProcessor()

    df = data.load_csv('../data/dataset_SCL.csv')
    syn_feat = data.generate_synthetic_features(df, ['Fecha-I', 'Fecha-O'])
    final_df = data.merge_dataframe([df, syn_feat])
    final_df = data.extract_and_prepare(final_df, target_data)
    features = data.encode_data(final_df)

    features_v1 = features.dropna()
    features_v1 = features_v1.drop(['OPERA', 'SIGLAORI', 'SIGLADES'], axis=1)
    X = features_v1.drop(['delay_15'], axis=1)
    y = features_v1['delay_15']
    X_train, X_test, y_train, y_test = data.split_and_oversample(X, y)

    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train,y_train)
    rf_pred = rf_model.predict(X_test)
    rf_pred_prob = rf_model.predict_proba(X_test)
    classification_summary(rf_pred, rf_pred_prob, y_test, 'Random Forest Classifier (RF)')

    pickle.dump(rf_model, open('../saved_model/rf_model.pickle', 'wb'))