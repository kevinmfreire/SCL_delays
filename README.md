# Flight Delays in Santiago de Chile Airport (SCL)

## Table of Content
* Overview
* Goals
* Methods
* Conclusion
* Future Work

## Overview
This project aims to develop a predictive model for the Santiago de Chile airport to determine the probability of a delay for a scheduled flight. The model will be built using historical flight data, and other relevant factors. The data will be preprocessed, analyzed, and formatted for use in machine learning algorithms such as decision trees, random forests, and neural networks. The model will be trained and validated using a combination of historical data. Once the model is developed and validated, it can be integrated into airport systems to provide real-time flight delay predictions to airlines, passengers, and airport personnel. The goal is to improve the efficiency of airport operations and enhance the overall travel experience for passengers.

## Goals
* Using 2017 Flight Data, preprocess, analyze and format data for the development of machine Learning models.
* Utilizing processed data, train and validate machine learning model with unseen historical data.

## Methods
* Processing Data:
    * Utilizing a combination of `matplotlib`, `seaborn`, and `pandas` analyze data.
    * Perform feature engineering to create additional features (see notebook for details).
    * With additional features analyze behaviour of delay accross the different season, months, days, etc.
    * Split dataset (70% training, 30% Testing)
    * With a unbalanced dataset (no delay > delay), utilize `SMOTE` technique to the training dataset in order to balance dataset.
* Training and validating data:
    * Training 4 different algorithms and performed A/B testing.
        * Logistic Regression
        * Decision Tree Classifier
        * Random Forest Classifier
        * Naive Bayes Classifier
    * Compared Performance performance of ML algorithms with 5 different metrics:
        * Accuracy
        * Precision
        * Recall
        * F1-score
        * AUC-ROC score
    * Made final decision based on top two metrics:
        * F1-score
        * AUC-ROC score

## Conclusion
The data was unbalanced having a larger amount of on-time flights than delayed flights so I used the SMOTE technique which is a form of oversampling the minority sample in order to have a balanced dataset.  Once I compared performance of the four ML algorithms my top performing model was the Random Forest Classifier.

## Future Work
I noticed that I've made a mistake in my training data.  I've included the Operation hour into my training data which may skew my model performance.  So when I have the chance I want to revisit this project and retrain it by removing a few training features.
