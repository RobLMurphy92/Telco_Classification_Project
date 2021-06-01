import pandas as pd
from scipy import stats
from pydataset import data
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression




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