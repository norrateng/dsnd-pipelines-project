# DSND Pipelines Project

## Background Context
In this project we are building a model to solve a binary classification problem, where we are predicting a binary outcome of whether a customer recommends a product based on their review that they have left. This includes data on the customer's age, written review, details on the product itself, and whether others found the review helpful.

## Summary
To start, we explore our data, examining aspects of the data such as its structure, whether it is balanced, and whether it is normally distributed. This allows us to make decisions on how we will transform our data, our choice of model, and any parameters used in the future. The data is then split into our train/test dataframes. We then build a pipeline which handles the different types of data, transforms them, and fits them to a model. We fine-tune and evaluate this model to find the best parameters to maximise the performance of our model. The model predicts the outcomes of whether a customer recommends the product.

#### Building pipeline
As part of the pipeline, each column in the dataframe undergoes a different pre-processing step depending on whether it is a numerical, categorical, or textual variable. 

For our numerical variables, the pipeline includes imputing our null values with its mean, and using the RobustScaler to standardise the numerical features. Although there are no null values in our data, we include this for completeness and reproducability in case there is an updated dataset. As our data is not normally distributed, the RobustScaler will be able to handle our data better as it is less sensitive to outliers.

For our categorical pipeline, our transformer also includes an imputer for the same reasons above, using the most frequent categories. It then includes a OneHotEncoder, which creates a binary output for each category of the categorical variables. Although this creates dimensionality for the data, it is appropriate as there is not ordinality with our categorical data.

Lastly, for our textual variables, the pipeline includes normalising, lemmatising, and vectorising the data. Through these three steps we have also tokenised the data and removed stop words. We combine the multiple textual variables together into one variable in our pipeline. This prepares the data and outputs one variable for each 'important word' in the textual fields with binary values.

#### Model selection
Model training and prediction is included in the pipeline - we test a Logistic Regression model and a Random Forest classifier and compare its outputs using metrics such as Accuracy, Precision, Recall, F1 Score, and ROC AUC. Our evaluation of the two models reveals that Logistic Regression outperforms Random Forest in most metrics, including accuracy, preciision, F1, and ROC AUC. Our dataset is also imbalanced, which means F1 score is the more important metric to consider. As recall is not as important for our use case, it makes sense to use Logistic Regression. 

#### Fine-tuning and evaluation
We fine-tune our Logistic Regression model by exploring which parameters can be tuned, and selecting options which are appropriate for our model using RandomizedSearchCV. The model is then re-run using the best parameters found and then re-evaluated. Our evaluation metrics show an uplift between our previous model and the best model, suggesting improvements to our fine-tuning process.


### Files 
- README.md - This is the README
- requirements.txt - This contains all the libaries you need to install in your environment to be able to run the notebook.
- starter.ipynb - This is the notebook which contains all the processes in this project.

### Set up instructions

Run requirements file:
```
python -m pip install -r requirements.txt
```