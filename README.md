# Machine-Learning-Heart-failure-Classification
This problem is a typical Classification Machine Learning task. We build various classifiers by using the following Machine Learning models such as: <br>
Logistic Regression (LR),<br> Decision Tree (DT), <br>Random Forest (RF) and <br>XGBoost (XGB).
# Data:
The dataset for exploration, modeling, and interpretability, explainability is called the "Heart Failure Clinical Records Data Set" to be found at the UCI (University of California Irvine) Machine Learning Repository.<br>
This dataset contains the medical records of 299 patients who had heart failure, collected during their follow-up period, where each patient profile has 13 clinical features.
Attribute Information - Thirteen clinical features: 
1. age: age of the patient (years)
2. anemia: decrease of red blood cells or hemoglobin (boolean)
3. high blood pressure: if the patient has hypertension (boolean)
4. creatinine phosphokinase (CPK): level of the CPK enzyme in the blood (mcg/L)
5. diabetes: if the patient has diabetes (boolean)
6. ejection fraction: percentage of blood leaving the heart at each contraction (percentage)
7. platelets: platelets in the blood (kiloplatelets/mL)
8. sex: woman or man (binary)
9. serum creatinine: level of serum creatinine in the blood (mg/dL)
10. serum sodium: level of serum sodium in the blood (mEq/L)
11. smoking: if the patient smokes or not (boolean)
12. time: follow-up period (days)
13. [target] death event: if the patient deceased during the follow-up period (boolean)
# Important Libraries used: (Apart from numpy, pandas, seaborn, matplotlib)
- Sklearn.ensemble RandomForestClassifier: A part of the Scikit-learn library, this is used for building a forest of decision trees for solving classification and regression problems. It's known for its high accuracy and ability to handle large datasets with higher dimensionality.
- Sklearn.feature_selection (SelectKBest, chi2, f_classif): These functions are used for feature selection to improve model performance. SelectKBest selects features according to the k highest scores, while chi2 and f_classif are scoring functions for classification tasks.
- Sklearn.linear_model LogisticRegression: A classification algorithm that is used to predict the probability of a categorical dependent variable.
- Sklearn.metrics: This module includes score functions, performance metrics, and pairwise metrics, and distance computations.
- Sklearn.model_selection: This module includes the model selection and evaluation tools.
- Sklearn.preprocessing RobustScaler: Scales features using statistics that are robust to outliers.
- Sklearn.tree DecisionTreeClassifier: A decision tree classifier for classifying data.
- Xgboost XGBClassifier: An implementation of gradient-boosted decision trees designed for speed and performance.
- Eli5: A library for debugging/inspecting machine learning classifiers and explaining their predictions.
- Lime (Local Interpretable Model-agnostic Explanations): A module for explaining machine learning classifiers with local surrogate models.
- Shap (SHapley Additive exPlanations): A game-theoretic approach to explaining the output of any machine learning model.
# Tasks performed:
1. Exploratory Data Analysis (EDA) to indicate how features correlate among themselves, with emphasis on the target/label one.
2. Machine Learning Modeling on the dataset using all four algorithms. Tune (hyper-parameter tuning) each model by calling the GridSearchCV method. Indicate which combination of Hyperparameters produces the best result. Note: Used accuracy and AUC-ROC metrics when evaluating the models.
3. Machine Learning Interpretability / Explanability tasks as follows: <br>
a. Use the 'eli5' library to interpret the "white box" model of Logistic Regression. Apply 'eli5' to visualize the weights associated with each feature. Use 'eli5' to explain specific predictions, pick a row in the test data with a negative label and one with a positive. <br>
b. Use the 'eli5' library to interpret the "white box" model of Decision Tree. Apply 'eli5' to list the feature importance ordered by the highest value. <br>
Get an explanation for a given prediction, one positive and one negative. This will calculate the contribution of each feature in the prediction. The explanation for a single prediction is calculated by following the decision path in the tree, and adding up the contribution of each feature from each node crossed into the overall probability predicted. eli5 can also be used to explain black box models, but we will use LIME and SHAP for our two last models instead. <br>
c. Use LIME to explain both the Random Forest and the XGBoost models. Create a LIME explainer by using the LimeTabularExplainer method, the main explainer to use for tabular data. LIME fits a linear model on a local shuffled dataset. Access the coefficients, the intercept, and the R2 of the linear model, for both model interpretability. <br>
Note: If R2 is low, the linear model that LIME fitted isn't a great approximation to your model, meaning you should not rely too much on the explanation it provides. <br>
d. Use the SHAP library to interpret the XGBoost model – specifically, TreeExplainer. This method of SHAP, TreeExplainer, is optimized for tree-based models. Visualize your explanations, one for positive and one for negative, using the ‘force_plot’ function. <br>
Note: You need to establish a ‘base value’ to be used by ‘force_plot’. The explainer.expected_value is the ‘base value’. Create the feature importance plot by calling SHAP’s ‘summary_plot’ function, for each class/label.
4. Predict observations, one for positive and one for negative labels, by using all four models and indicate which one gives the better prediction. Provide output for showing the accuracy of each model as follows: False/True label: 0/1 (or 0/1 depending on how we define labels)
- LR: [prob_T prob_F]
- DT: [prob_T prob_F]
- RF: [prob_T prob_F]
- XGB: [prob_T prob_F] <br>
The above calculations are derived by calling the predict_proba method. Note: predict_proba(X): Predict class probabilities for X.
# Conclusion:
In this machine learning project, we embarked on a journey to develop predictive models for heart failure using a dataset comprising the medical records of 299 patients. Our goal was to create accurate models while gaining insights into the factors influencing these predictions. <br>
## Summarize the findings:
### Model Performance
We applied a range of classification algorithms, including Logistic Regression, Decision Tree, Random Forest, and XGBoost, and achieved impressive accuracy scores, consistently in the range of 80% to 86%. These models showed their potential in predicting heart failure.
### Feature Importance
Model interpretability was a key focus. We leveraged tools like ELI5, LIME, and SHAP to delve into feature importance. We observed that factors like serum_creatinine, time, age, and ejection_fraction played significant roles in predicting heart failure, providing valuable medical insights.
### Local Interpretability
LIME enabled us to understand individual predictions better. For instance, we discovered that for specific cases, higher time values indicated a reduced risk of heart failure, while elevated serum_creatinine levels raised concerns.
### Global Interpretability
SHAP analysis sheds light on the global impact of features. We noted that serum_sodium and high_blood_pressure contributed negatively to heart failure predictions, while time and serum_creatinine had significant positive impacts.
### Model Comparison
A thorough comparison of models highlighted their unique strengths. Decision Trees showed perfect prediction for some instances but were prone to overfitting. Random Forest and XGBoost excelled in generalization and performed consistently well. <br>
In conclusion, our project has demonstrated the potential of machine learning in predicting heart failure while emphasizing model interpretability. We've unveiled crucial factors influencing predictions and established a foundation for responsible and transparent predictive modeling in healthcare.
