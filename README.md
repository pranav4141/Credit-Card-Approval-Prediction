![banner](assets/Credit_card_approval_banner.png)
Banner [source](https://banner.godori.dev/)

![Python version](https://img.shields.io/badge/Python%20version-3.10%2B-lightgrey)
![Type of ML](https://img.shields.io/badge/Type%20of%20ML-Binary%20Classification-red)
[![Open Source Love svg1](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)

Badge [source](https://shields.io/)

# Description: A machine learning project that predicts whether a credit card applicant is likely to be classified as high risk based on applicant information. The project includes exploratory data analysis, feature engineering, preprocessing, comparison of multiple classification models, final model testing, and a Streamlit interface.

## Authors

- [@pranav4141](https://www.github.com/pranav4141)

## Table of Contents

  - [Business problem](#business-problem)
  - [Data source](#data-source)
  - [Methods](#methods)
  - [Tech Stack](#tech-stack)
  - [Quick glance at the results](#quick-glance-at-the-results)
  - [Lessons learned and recommendation](#lessons-learned-and-recommendation)
  - [Limitation and what can be improved](#limitation-and-what-can-be-improved)
  - [Run Locally](#run-locally)
  - [Explore the notebook](#explore-the-notebook)
  - [Deployment on streamlit](#deployment-on-streamlit)
  - [App deployed on Streamlit](#app-deployed-on-streamlit)
  - [Repository structure](#repository-structure)
  - [Contribution](#contribution)
  - [License](#license)




## Business problem
Each time there is a hard enquiry your credit score is affected negatively. This app predict the probability of being approved without affecting your credit score. This project predicts whether a credit card applicant is likely to be classified as high risk based on their application information.
## Data source

- [Kaggle credit card approval prediction](https://www.kaggle.com/rikdifos/credit-card-approval-prediction)

## Methods

- Exploratory data analysis
- Feature Engineering
- Multivarate correlation
- SMOTE Oversampling
- Select Gradient Boosting
- Model deployment
  
## Tech Stack
- Python (refer to requirement.txt for the packages used in this project)
- Streamlit (interface for the model)
- Data Analysis: Pandas, NumPy, Matplotlib, Seaborn
- Machine Learning: Scikit-learn, Imbalanced-learn, SMOTE, Gradient Boosting, Logistic Regression, Random Forest, Support Vector Machine, Decision Tree, AdaBoost
- Deployment: Streamlit, Github
- Supporting Libraries: Joblib, Requests, SciPy


## Quick glance at the results

Correlation between the features.

![heatmap](assets/heatmap.png)

Confusion matrix of gradrient boosting classifier.

![Confusion matrix](assets/confusion_matrix.png)

ROC curve of gradrient boosting classifier.

![ROC curve](assets/roc.png)

Top 3 models (with default parameters)

| Model     	                | Recall score 	|
|-------------------	        |------------------	|
| Support vector machine     	| 88% 	            |
| Gradient boosting    	        | 90% 	            |
| Adaboost               	    | 79% 	            |


- ***The final model used is: Gradient boosting***
- ***Metrics used: Recall***

### Why Recall?

Recall is important because the project focuses on identifying applicants belonging to the high-risk class. A higher recall means the model is better at identifying actual high-risk applicants.

There is a trade-off between precision and recall, so both metrics should be considered when evaluating the model.

## Lessons learned and recommendation

- Class imbalance can significantly affect classification performance, making techniques such as SMOTE useful.
- Feature engineering and preprocessing have an important impact on model performance.
- Different classification algorithms can perform differently on the same dataset.
- Gradient Boosting provided the strongest recall among the compared models.
- Recall and precision should be considered together when evaluating classification models for risk-related problems.
- A machine learning model should be evaluated on unseen test data rather than relying only on training performance

## Limitations and Future Improvements
-The dataset is historical and may not represent current credit approval practices. 
-The model should be retrained and validated using more recent data before any real-world application.
-More extensive hyperparameter tuning could potentially improve model performance.
-The preprocessing and model could be combined into a single fitted pipeline for more reliable deployment.
-Probability calibration and decision-threshold tuning could be explored.
-Fairness and bias evaluation would be important before applying the model to real financial decisions.
-The first Streamlit startup can take some time because the model is trained from the dataset. Streamlit caching is used to avoid unnecessary retraining during subsequent reruns.


## Run Locally

Clone the project

```bash
git clone https://github.com/pranav4141/Credit-Card-Approval-Prediction.git
```
enter the project directory

```bash
cd Credit-Card-Approval-Prediction
```

```bash
pip install -r requirements.txt
```
Start the streamlit server locally

```bash
streamlit run cc_approval_pred.py
```
If you are having issue with streamlit, please follow [this tutorial on how to set up streamlit](https://docs.streamlit.io/library/get-started/installation)

## Explore the notebook

To explore the notebook file [here](https://github.com/pranav4141/Credit-Card-Approval-Prediction/blob/main/Credit_card_approval_prediction.ipynb)

## Deployment on streamlit

To deploy this project on streamlit share, follow these steps:

- first, Push the project files to GitHub.
- Make sure requirements.txt is in the repository root.
- Open Streamlit Community Cloud.
- Connect your GitHub account.
- Select this repository.
- Select the appropriate branch.
- Set the main file to: cc_approval_pred.py
- Deploy the application.

## Repository structure


```

├── assets
│   ├── Credit_card_approval_banner.png       <- banner image used in the README.
│   ├── confusion_matrix.png                  <- confusion matrix image used in the README.
│   ├── environment.yml                       <- list of all the dependencies with their versions(for conda environment).
│   ├── heatmap.png                           <- heatmap image used in the README.
│   ├── roc.png                               <- ROC image used in the README.
|
├── datasets
│   ├── application_record.csv                <- the application record data.
│   ├── credit_record.csv                     <- the credit record data.
│   ├── test.csv                              <- the test data.
│   ├── train.csv                             <- the train data.
│
├── .gitignore                                <- used to ignore certain folder and files that won't be commit to git.
│
│
├── Credit_card_approval_prediction.ipynb     <- main python notebook where all the analysis and modeling are done.
│
│
├── README.md                                 <- this readme file.
│
│
├── cc_approval_pred.py                       <- file with the best model and best hyperparameter with streamlit component for rendering the interface.
│
│
├── requirements.txt                           <- list of all the dependencies with their versions(used for Streamlit ).

```
