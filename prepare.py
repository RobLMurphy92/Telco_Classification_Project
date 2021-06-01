import seaborn as sns
import os
from pydataset import data
from scipy import stats
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

# ignore warnings

# ignore warnings
import warnings
from acquire import get_telco_churn_data
warnings.filterwarnings("ignore")




    


###################### Prep Telco Data ######################

def telco_split(df):
    '''
    This function take in the telco_churn data acquired by get_telco_data,
    performs a split and stratifies churn column.
    Returns train, validate, and test dfs.
    '''
    #20% test, 80% train_validate
    train_validate, test = train_test_split(df, test_size=0.2, 
                                        random_state=1349, 
                                        stratify=df.churn)
    # 80% train_validate: 30% validate, 70% train.
    train, validate = train_test_split(train_validate, train_size=0.7, 
                                   random_state=1349, 
                                   stratify=train_validate.churn)
    return train, validate, test




def prep_telco(df):
    '''
    This function take in the telco_churn data acquired by get_telco_data,
    Returns prepped train, validate, and test dfs with embarked dummy vars,
    deck dropped, and the mean of age imputed for Null values.
    '''
    # drop duplciates
    df.drop_duplicates(inplace =True)
    # dropped duplicate columns.
    df = df.loc[:, ~df.columns.duplicated()]
    #Changing the index from numbered records to records by customer_id
    df.set_index('customer_id', drop = True, inplace = True)
    #checking for empty records, replaced the empty values with NaN
    df.replace(' ', np.nan, inplace = True)
    #dropped the nan values within the total charges column
    df.dropna(inplace =True)
    #total_charges is a dtype object, but needs to be changed to a float64.
    df = df.astype({'total_charges': 'float64'})
    #creating dummies on categorical variables which are non-binary.
    df = create_dummies(df)
    
    #replacing several categorical fields str values to numerical based values.
    df.gender = df.gender.replace({'Female': 1, 'Male':0})
    df.partner = df.partner.replace({'Yes': 1, 'No':0})
    df.dependents = df.dependents.replace({'Yes': 1, 'No':0})
    df.churn = df.churn.replace({'Yes': 1, 'No':0})
    df.phone_service= df.phone_service.replace({'Yes': 1, 'No':0})
    df.paperless_billing = df.paperless_billing.replace({'Yes': 1, 'No':0})
    
    #dropping original fields and renaming fields.
    df.drop(columns = ['multiple_lines','online_security', 'online_backup', 'device_protection', 'tech_support','streaming_tv', 
                                'streaming_movies','paperless_billing','contract_type_id','paperless_billing', 'payment_type_id','payment_type'], inplace =True)
    df.rename(columns = {'tenure': 'tenure_months', 'Bank transfer (automatic)': 'bank_transfer_auto', 'Credit card (automatic)': 'credit_card_auto',
                                                'Mailed check': 'mailed_check','Fiber optic': 'fiber_optic', 'Electronic check': 'electronic_check', 'Two year': 'two_year_contract', 'Month-to-month': 'Month-to-month_contract', 'One year': 'one_year'}, inplace = True)
    df['auto_pay'] = df['bank_transfer_auto'] + df['credit_card_auto']
    df.drop(columns = ['bank_transfer_auto', 'credit_card_auto', 'mailed_check','electronic_check'], inplace = True)
    #created a new field to change tenure from months to year.
    df['tenure_year'] =  round(df['tenure_months']/12,0)
    
    return df
    
  
    
##############################################################################
def create_dummies(df):
    '''
    This function is used to create dummy columns for my non binary columns
    '''
    # create dummies for payment_type, internet_service_type, and contract_type
    payment_dummies = pd.get_dummies(df.payment_type, drop_first=False)
    internet_dummies = pd.get_dummies(df.internet_service_type, drop_first=False)
    contract_dummies = pd.get_dummies(df.contract_type, drop_first=False)

    # now we concatenate our dummy dataframes with the original
    df = pd.concat([df, payment_dummies, internet_dummies, contract_dummies], axis=1)
    df = df.rename(columns = {'None': 'no_internet'})
    

    # now I am dropping all my original string columns that I made dummies with and dropping
    #the type_id columns since they are duplicates of the string column
    df = df.drop(columns=['contract_type', 'internet_service_type', 'no_internet'])
    return df





##############################################################################
def model_metrics(X_train, y_train, X_validate, y_validate):
    '''
    this function will score models and provide confusion matrix.
    returns classification report as well as evaluation metrics.
    '''
    lr_model = LogisticRegression(random_state =1349)
    dt_model = DecisionTreeClassifier(max_depth = 2, random_state=1349)
    rf_model = RandomForestClassifier(max_depth=4, min_samples_leaf=3, random_state=1349)
    models = [lr_model, dt_model, rf_model]
    for model in models:
        #fitting our model
        model.fit(X_train, y_train)
        #specifying target and features
        train_target = y_train
        #creating prediction for train and validate
        train_prediction = model.predict(X_train)
        val_target = y_validate
        val_prediction = model.predict(X_validate)
        # evaluation metrics
        TN_t, FP_t, FN_t, TP_t = confusion_matrix(y_train, train_prediction).ravel()
        TN_v, FP_v, FN_v, TP_v = confusion_matrix(y_validate, val_prediction).ravel()
        #calculating true positive rate, false positive rate, true negative rate, false negative rate.
        tpr_t = TP_t/(TP_t+FN_t)
        fpr_t = FP_t/(FP_t+TN_t)
        tnr_t = TN_t/(TN_t+FP_t)
        fnr_t = FN_t/(FN_t+TP_t)
        tpr_v = TP_v/(TP_v+FN_v)
        fpr_v = FP_v/(FP_v+TN_v)
        tnr_v = TN_v/(TN_v+FP_v)
        fnr_v = FN_v/(FN_v+TP_v)
        
        
        
        print('--------------------------')
        print('')
        print(model)
        print('train set')
        print('')
        print(f'train accuracy: {model.score(X_train, y_train):.2%}')
        print('classification report:')
        print(classification_report(train_target, train_prediction))
        print('')
        print(f'''
        True Positive Rate:{tpr_t:.2%},  
        False Positive Rate :{fpr_t:.2%},
        True Negative Rate: {tnr_t:.2%},  
        False Negative Rate: {fnr_t:.2%}''')
        print('------------------------')
        
        print('validate set')
        print('')
        print(f'validate accuracy: {model.score(X_validate, y_validate):.2%}')
        print('classification report:')
        print(classification_report(y_validate, val_prediction))
        print('')
        print(f'''
        True Positive Rate:{tpr_v:.2%},  
        False Positive Rate :{fpr_v:.2%},
        True Negative Rate: {tnr_v:.2%},  
        False Negative Rate: {fnr_v:.2%}''')
        print('')
        print('------------------------')
        
    
 


    
    
    