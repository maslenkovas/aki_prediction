# Prediction of Acute Kidney Injury using Electronic Health Records.
The deep learning model for predicting Acute Kidney Injury (AKI) in ICU patients from Electronic Health Records (EHR) database MIMIC IV v2.0.

The database is publicly available on [Physionet MIMIC-IV](https://physionet.org/content/mimiciv/2.0/) page.

- File icu_preprocessing.ipynb is used to preproccess the data into the format suitable for the model.
- File icu_lstm_sigmoid.py contains all functions and modules for training and evaluating the model.
- File Cohort_analysis.ipynb contains initial analyses of the data used to train the model. 
- File icu_xgb.ipynb is used to train and evaluate XGBoost AKI prediction model. Preprocessing pipeline for XGBoost is located in the icu_preprocessing.ipynb file.
- File tokenizer.json is a trained tokenizer used in the model training.

Please let me know if you have any questions by the folloving email: maslenkova.lana@gmail.com
