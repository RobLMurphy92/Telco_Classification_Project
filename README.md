# Telco_Classification_Project
<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>



#### Project Objectives
> - Document code, process (data acquistion, preparation, exploratory data analysis and statistical testing, modeling, and model evaluation), findings, and key takeaways in a Jupyter Notebook report.
> - Create modules (acquire.py, prepare.py) that make your process repeateable.
> - Construct a model to predict customer churn using classification techniques.
> - Deliver a 5 minute presentation consisting of a high-level notebook walkthrough using your Jupyter Notebook from above; your presentation should be appropriate for your target audience.
> - Answer panel questions about your code, process, findings and key takeaways, and model.


#### Business Goals
> - Find drivers for customer churn at Telco.
> - Construct a ML classification model that accurately predicts customer churn.
> - Document your process well enough to be presented or read like a report.

#### Audience
> - Your target audience for your notebook walkthrough is the Codeup Data Science team. This should guide your language and level of explanations in your walkthrough.

#### Project Deliverables
> - a Jupyter Notebook Report showing process and analysis with the goal of finding drivers for customer churn. This notebook should be commented and documented well enough to be read like a report or walked through as a presentation.
> - a README.md file containing the project description with goals, a data dictionary, project planning (lay out your process through the data science pipeline), instructions or an explanation of how someone else can recreate your project and findings (What would someone need to be able to recreate your project on their own?), key findings, recommendations, and takeaways from your project.
> - a CSV file with customer_id, probability of churn, and prediction of churn. (1=churn, 0=not_churn). These predictions should be from your best performing model ran on X_test. Note that the order of the y_pred and y_proba are numpy arrays coming from running the model on X_test. The order of those values will match the order of the rows in X_test, so you can obtain the customer_id from X_test and concatenate these values together into a dataframe to write to CSV.
individual modules, .py files, that hold your functions to acquire and prepare your data.
> - a notebook walkthrough presentation with a high-level overview of your project (5 minutes max). You should be prepared to answer follow-up questions about your code, process, tests, model, and findings.


<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

#### Data Dictionary

|Target|Datatype|Definition|
|:-------|:--------|:----------|
| churn | 150 non-null: int64 | churned is a value of 1, not churned is value of 0 |
|Feature|Datatype|Definition|
|:-------|:--------|:----------|
|total_charges| 7032 non-null: float64 |    overall charges for services in currency |
|monthly_charges| 7032 non-null: float64 |    charges for services in currency|
|auto_pay| 7032 non-null: uint8|    0 for does not have auto_pay, 1 for has auto_pay|
|two_year_contract| 7032 non-null: uint8 |    0 for does not have two_year, 1 for has two_year|
|fiber_optic| 7032 non-null: uint8 |    0 for does not have fiber, 1 for has fiber |
|Month-to-month_contract| 7032 non-null: uint8 |    0 for not monthly, 1 for is monthly |



<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

#### Project Planning:

> - Create a README.md which contains a data dictionary, project objectives, business goals, initial hypothesis.
> - Acquire the telco dataset from the Codeup databse, create a function which will use a sql query and pull specific tables save this function in a acquire.py
> - Prep the telco dataset and clean it as well, remove unwanted data, make alterations to the datatypes.
> - Change any categorical variables into a binary categorical. Create a function to simplify this process and include it in a prepare.py
> - Calculate your baseline accuracy and use this for comparing adequacy of the model.
> - Train three different classification models.
> - Evaluate the models on the train and validate datasets.
> - Choose the model which performs the best, then run that model on the test dataset.
> - Create csv file.
> - Present conclusions and main takeaways.

#### Initial Hypotheses:

> - **Hypothesis 1 -** I rejected the Null Hypothesis; there is a difference.
> - alpha = .05
> - $H_0$: Sepal length is the same in virginica and versicolor. $\mu_{virginica} == \mu_{versicolor}$.  
> - $H_a$: Sepal length significantly different in virginica and versicolor. $\mu_{virginica} != \mu_{versicolor}$. 

> - **Hypothesis 2 -** I rejected the Null Hypothesis; there is a difference.
> - alpha = .05
> - $H_0$: Sepal width is the same in virginica and versicolor. $\mu_{virginica} == \mu_{versicolor}$.  
> - $H_a$: Sepal width significantly different in virginica and versicolor. $\mu_{virginica} != \mu_{versicolor}$. 




### Reproduce My Project:

<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

You will need your own env file with database credentials along with all the necessary files listed below to run my final project notebook. 
- [x] Read this README.md
- [ ] Download the aquire.py, prepare.py, explore.py, model.py and final_notebook.ipynb files into your working directory
- [ ] Add your own env file to your directory. (user, password, host)
- [ ] Run the final_notebook.ipynb 

