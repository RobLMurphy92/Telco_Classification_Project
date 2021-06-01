# Telco_Classification_Project


Project Planning

Create a README.md which contains a data dictionary, project objectives, business goals, initial hypothesis.
Acquire the telco dataset from the Codeup databse, create a function which will use a sql query and pull specific tables save this function in a acquire.py
Prep the telco dataset and clean it as well, remove unwanted data, make alterations to the datatypes.
Change any categorical variables into a binary categorical. Create a function to simplify this process and include it in a prepare.py
Calculate your baseline accuracy and use this for comparing adequacy of the model.
Train three different classification models.
Evaluate the models on the train and validate datasets.
Choose the model which performs the best, then run that model on the test dataset.
Create csv file.
Present conclusions and main takeaways.


Project Goal:
Find drivers for customer churn at Telco.
Construct a ML classification model that accurately predicts customer churn.
Document your process well enough to be presented or read like a report.

Audience:
Your target audience for your notebook walkthrough is the Codeup Data Science team. This should guide your language and level of explanations in your walkthrough.


Deliverables:
You are expected to deliver the following:

a Jupyter Notebook Report showing process and analysis with the goal of finding drivers for customer churn. This notebook should be commented and documented well enough to be read like a report or walked through as a presentation.
a README.md file containing the project description with goals, a data dictionary, project planning (lay out your process through the data science pipeline), instructions or an explanation of how someone else can recreate your project and findings (What would someone need to be able to recreate your project on their own?), key findings, recommendations, and takeaways from your project.
a CSV file with customer_id, probability of churn, and prediction of churn. (1=churn, 0=not_churn). These predictions should be from your best performing model ran on X_test. Note that the order of the y_pred and y_proba are numpy arrays coming from running the model on X_test. The order of those values will match the order of the rows in X_test, so you can obtain the customer_id from X_test and concatenate these values together into a dataframe to write to CSV.
individual modules, .py files, that hold your functions to acquire and prepare your data.
a notebook walkthrough presentation with a high-level overview of your project (5 minutes max). You should be prepared to answer follow-up questions about your code, process, tests, model, and findings.


Data dictionary:


Project planning:
Create a README.md which contains a data dictionary, project objectives, business goals, initial hypothesis.
Acquire the telco dataset from the Codeup databse, create a function which will use a sql query and pull specific tables save this function in a acquire.py
Prep the telco dataset and clean it as well, remove unwanted data, make alterations to the datatypes.
Change any categorical variables into a binary categorical. Create a function to simplify this process and include it in a prepare.py
Calculate your baseline accuracy and use this for comparing adequacy of the model.
Train three different classification models.
Evaluate the models on the train and validate datasets.
Choose the model which performs the best, then run that model on the test dataset.
Create csv file.
Present conclusions and main takeaways.


Initial ideas/hypotheses stated:


Instructions for recreating project/running repo:

