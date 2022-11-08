# Prediction of Acute Kidney Injury using Electronic Health Records.
The deep learning model for predicting Acute Kidney Injury (AKI) in ICU patients from Electronic Health Records (EHR) database MIMIC IV v2.0.

## Project description
Acute Kidney Injury (AKI) affects more than 13 million people annually and increases
the risk of death in patients[26]. The severity of AKI also contributes to the increase
in associated costs of a patient’s treatment. The early prediction of AKI could enable
clinicians to focus on preventive treatment for at-risk patients.

In addition to the classical machine learning approach called The extreme gradient boosting (XGBoost) with manual feature selection,
we have also explored an LSTM-based approach applied to the prediction of AKI. Due to
the variety of data available for each patient, it is challenging to assess which information
could be the best predictor. Thus, the text classification model used unstructured textual
data to make predictions.

![Alt text](aki_prediction/images/Architechture.jpg?raw=true "Title")

## Files description
The database is publicly available on [Physionet MIMIC-IV](https://physionet.org/content/mimiciv/2.0/ | width=100) page.

- File icu_preprocessing.ipynb is used to preproccess the data into the format suitable for the model.
- File icu_lstm_sigmoid.py contains all functions and modules for training and evaluating the model.
- File Cohort_analysis.ipynb contains initial analyses of the data used to train the model. 
- File icu_xgb.ipynb is used to train and evaluate XGBoost AKI prediction model. Preprocessing pipeline for XGBoost is located in the icu_preprocessing.ipynb file.
- File tokenizer.json is a trained tokenizer used in the model training.

Please let me know if you have any questions by the folloving email: maslenkova.lana@gmail.com
