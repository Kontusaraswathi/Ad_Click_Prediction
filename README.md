# Ad_Click_Prediction
This project predicts whether a user will click on an online advertisement based on various features such as age, daily internet usage, area income, and more.
It is built using Python and Machine Learning algorithms for classification

📜 Overview
Online advertising is one of the primary sources of revenue for many businesses.
Being able to predict whether a user will click on an ad can help companies optimize ad placement and targeting.


In this project, we:
Preprocess the dataset.
Perform Exploratory Data Analysis (EDA).
Train multiple classification models.
Evaluate and compare their performance.
Deploy the best model.


📂 Dataset
The dataset contains user information such as:
Daily Time Spent on Site
Age
Area Income
Daily Internet Usage
Ad Topic Line
City
Male (Gender)


Clicked on Ad (Target Variable)
📌 Target variable: Clicked on Ad
1 → User clicked on the ad
0 → User did not click


You can download a sample dataset from:
Ad Click Prediction Dataset: https://www.kaggle.com/datasets/marius2303/ad-click-prediction-dataset

Technologies Used
Python 3.8+
Pandas, NumPy
Matplotlib, Seaborn
Scikit-learn
Jupyter Notebook


Installation
Clone the repository and install dependencies:
git clone https://github.com/Kontusaraswathi/Ad_Click_Prediction.git
cd ad-click-prediction
pip install -r requirements.txt


Model Training & Evaluation
Models tested:
Logistic Regression
Random Forest Classifier
Decision Tree Classifier
Gradient Boosting

Evaluation Metrics:
Accuracy
Precision
Recall
F1 Score

Example:
Model	Accuracy	Precision	Recall	F1 Score
Logistic Regression	91%	90%	89%	89%
Random Forest	94%	93%	92%	92%

✅ Results
Best Model: Random Forest Classifier

Accuracy Achieved: 94%

Insights: Age, Daily Internet Usage, and Area Income are key factors affecting ad clicks.

🚀 Future Improvements
Use deep learning models for comparison.

Include more behavioral features.

Deploy as a web app using Flask or Streamlit.
