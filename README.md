**Capstone Project**  
**Customer Churn Prediction**  
**Final Report**

**1.	Problem Statement**
   
The problem is how to identify at an early stage the customers who are likely to churn so that proactive measures can be implemented to prevent these customers from leaving. 

We would like to predict whether a customer will churn or not churn based on certain attributes like customer satisfaction scores, how long the customer has been with the company, payment failures, customer location, customer engagement level and the contract type.

It is critical to find the reasons why customers stop using the company's products. When these factors are known, the customers who are likely to churn can be identified early enough and special programs can be implemented to prevent these customers from churning. In this way, the company's continued profitability is assured.

Customer retention makes a lot of business sense - it costs around five (5) times more to acquire new customers than to retain existing customers. A reduction in customer churn can significantly increase revenue. The reasons for customer churn and who are these customers can be predicted before churn happens.

Engaging with unhappy customers, resolving issues and building stronger relationships directly impact customer retention.

Success can be measured in terms of a reduction in churn rate. Also, a Profit/Loss analysis is made by considering the increase in revenue against the money spent for customer retention.

The models that are produced can be used by marketing teams for retention campaigns, by customer support teams for proactive outreach, and by product teams for feature improvements. Leadership can also be guided by the model for churn strategy and forecasting.
The reasons for customer churn and who are these customers can be predicted before churn happens.

**2.	Model Outcomes or Predictions**
   
The type of learning is classification. The expected output of the selected model is the prediction of customers who are likely to churn and the features that drive churn. Supervised machine learning algorithms are used to build predictive models. 

The models are expected to be able to catch churners quite well so that proactive measures can be designed at an early stage to prevent them from leaving.  At the same time, a profit/loss analysis makes sure that these measures are profitable. The models will also demonstrate a capability to make useful predictions and highlight the features mostly affecting churn so that retention programs are suitably targeted.

**3.	Data**
   
The `Customer Churn Prediction Business Dataset` comes from `Kaggle`. This dataset is synthetically generated for educational, research, and portfolio purposes. While it reflects realistic business patterns, it does not represent real customer data.

The less obvious meaning of some features in the data are as follows: 

‘csat_score’ is Customer Satisfaction Score that measures short-term happiness with a specific event. 

'nps_score' is Net Promoter Score that measures long term loyalty. 

'email_open_rate' indicates if customers find subject lines and sender information compelling enough to open the mail. Declining rates of these features predict disinterest and potential churn. High open rates suggest relevance and connection, while low rates flag a need to improve content, segmentation or timing to prevent users from becoming disengaged and leaving.

**4.	Data Preprocessing/Preparation**
   
The `'customer_id'` column is dropped because it does not add value to the modeling effort:  
`df.drop(columns=['customer_id'])`  

Any leading and trailing white spaces from categorical columns are removed:  
`string_cols = df.select_dtypes(include=['object']).columns`  
`df[string_cols] = df[string_cols].apply(lambda x: x.str.strip())`  

The column with missing values is identified:   
`df.columns[df.isnull().any()].tolist()`    

The percentage of missing values is calculated:   
`df['complaint_type'].isnull().sum()/df.shape[0]*100`     

The percentage of missing values is not high enough to warrant dropping the column. The missing values are instead replaced with the column mode:    
`mode_column = df['complaint_type'].mode()`  
`df['complaint_type'] = df['complaint_type'].fillna(str(mode_column))`

Inconsistent data is replaced, for example:   
`df['complaint_type'] = df['complaint_type'].replace({'0    Technical\nName: complaint_type, dtype: object':'Technical'})`  

There are no duplicate rows:  
`duplicates = len(df[df.duplicated()])`  

The distributions of categorical columns are verified:  
`categorical_cols = df.select_dtypes(include=['object']).columns.tolist()`  
`for i in categorical_cols:`  
 `   value_count_column = df[i].value_counts(normalize=True)`  
 `   print(f'The value count for column {value_count_column} \n')`   

Box plots of numerical features are analyzed prior to removal of outliers to visualize the presence of outliers.  

Outliers are removed from numeric features using the interquartile range (IQR) rule:  
`for col in numeric_list:`  
`    Q1 = df[col].quantile(0.25)`  
`    Q3 = df[col].quantile(0.75)`  
`    IQR = Q3 - Q1`  
`    lower_bound = Q1 - 1.5*IQR`  
`    upper_bound = Q3 + 1.5*IQR`  
`    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]`  

Box plots of numerical features are re-analyzed after the removal of outliers. The remaining outliers are allowed to stay to conserve data.   

For the box plots, particular attention is given to when the `‘churn’` median line is lower than the `‘no churn’` median line, churn is more likely here. Overlapping `‘churn’` and `‘no churn’` boxes indicate that the feature might not be a good predictor. Separation of the churn and no churn boxes is a strong signal for good predictors.  

Stacked bars of categorical features (in percentages) are plotted.    The color ratios of the stacked bars highlight features which have slightly more churn.  

Histograms are plotted to visualize the distribution of numerical features to help uncover trends and patterns.  

Heat maps of numeric features are displayed to highlight significant positive and negative correlations especially with `‘churn’`.  

`total_revenue` is positively correlated with `tenure_months` (0.62) and `monthly_fee` (0.60) which is considered high. `total_revenue` does not add new predictive signal beyond `tenure_months` and `monthly_fee` since `total_revenue` is the product of `tenure_months` and `monthly_fee`. This means that `total_revenue` is mostly redundant and is, therefore, dropped. 

Feature engineering is applied by creating a new feature (`tenure_fee_interaction`) which is a product of `tenure_months` and `monthly_fee`.  
`df['tenure_fee_interaction'] = df['tenure_months'] * df['monthly_fee']`
`df = df.drop(columns=['total_revenue'])`

A pair plot analysis of the top six (6) numeric features is made to verify the distribution of churn in these numeric features.   

The features and target variable are defined:  
`X = df.drop(['churn'], axis=1)`  
`y = df['churn']`  

and then split into training and test sets:  
`X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)`  

Scaling is not used in `DecisionTreeClassifier` while  `LogisticRegression`, `KNeighborsClassifier` and `SVC (Support Vector Classifier)` use scaling. `StandardScaler` is used. `KNeighborsClassifier` and `SVC (Support Vector Classifier)`are especially sensitive to the scale of input features.  

Categorial features are encoded using `OneHotEncoder` and `OrdinalEncoder`.

**5.	Modeling**
   
Seven (7) supervised machine learning classification algorithms are used to build predictive models:  
`LogisticRegression, DecisionTreeClassifier, KNeighborsClassifier, SVC, Random Forest Classifier, Keras Classifier,` and `XGBoost Classifier`. 

The class imbalance is verified:  
`val_count_churn = df['churn'].value_counts(normalize=True)`  
`print(val_count_churn)`  
`churn`  
`0    0.906262`  
`1    0.093738`  
`Name: proportion, dtype: float64`  

In order to address the class imbalance, `SMOTE-NC (Synthetic Minority Oversampling Technique for Nominal and Continuous features)`is used by `KNeighborsClassifier` while the other models use the parameter `class_weight='balanced'`.  

Transformers like `make_column_transformer` and `ColumnTransformer` are used to prepare the data for encoding and scaling, as required, and fed to a `Pipeline`.  

A baseline model is built to benchmark the models to be designed. Simple models of the various algorithms are initially created to further benchmark the modeling effort.  

`classification_report` and `confusion_matrix` provide the `Accuracy`, `Precision`, and `Recall` of the models. `F2 score` is calculated and a `Profit/Loss analysis` is also made.   

The models are optimized using `HalvingRandomSearchCV`, `GridSearchCV`, and `RandomizedSearchCV`.  

The `ROC (Receiver Operating Characteristic) Curve` is plotted to display the  `AUC (Area Under Curve)` which is a measure of the predictive power of the model and to calculate the optimal threshold using the `Youden’s J` method.  

The `Precision-Recall Curve` is also plotted to demonstrate the relationship between `Precision` and `Recall`. `Precision` and `Recall` are evaluated at different thresholds.  The threshold that maximizes profit is determined.  

For each classifier, models are constructed and their respective performances are compared to each other. In addition to `Accuracy`, `Precision`, `Recall`, `F2 score`, and `AUC`, the best model is chosen based on business goals that consider the relative cost of missing a churner (`False Negatives` - predicted not to churn but churned, loss of lifetime value), cost of false alarms (`False Positives` - predicted to churn but stayed, cost of retention offer) and `True Positives` (predicted to stay and actually stayed, saved lifetime value less the cost of retention offer).  

In this particular case of customer churn, missing churners (`Recall`) is more expensive than false alarms (`Precision`). `Recall` is, therefore, optimized at the expense of `Precision` and `Accuracy`.  
The features and their proportional importances are also identified particularly those features whose increase or decrease correspondingly raise or lower the likelihood of customer churn.  

Prediction is demonstrated using samples from the test data.  

Identification of customers likely to churn is also shown.      

A `Churn Risk Demonstration List` is taken from the test data to demonstrate one of the many ways to use the model. The list shows customers who have churned and those who have not. We need to pay close attention to those customers who have not churned yet and are at risk of churning.      
For example, we shall identify customers with high churn probability (or showing signs of churning) and who have not yet churned so that we can embark on special and proactive programs to keep them from leaving.

Model hyperparameters are optimized using grid search to maximize model performance particularly addressing overfitting.

Feature selection of encoded features reduces the number of features by around 40% to improve the model and to lessen noise. When there is less noise, the interpretability of feature importance improves. Noise in this context is irrelevant information that obscures underlying patterns or relationships the model is trying to learn.            

The features and their importances are identified particularly those features whose increase or decrease correspondingly raise or lower the likelihood of customer churn. The features and their importances are determined to verify the magnitude of features influence.      

After feature selection, the results of feature importance from the top three (3) models are fed to a Generative AI (Gen AI) model for interpretation. The codes are commented because the API key can only be accessed locally. However, the Gen AI narrative is captured and shown.


**6.	Model Evaluation**
   
The GenAI model provided the following feature importance analysis on the results of the top three (3) classification models which are averaged.  There are slight changes between runs and they usually come from randomness in the pipeline but the ranking is generally maintained.

"Based on the feature importance data, here are some key insights and observations: Top Important Features: 1. csat_score (0.884): The customer satisfaction score is the most important feature, indicating that customer satisfaction is a critical factor in predicting customer churn. This suggests that addressing customer concerns and improving overall customer experience can help retain customers. 2. tenure_months (0.882): Customer tenure is the second most important feature, indicating that longer-term customers are more likely to churn. This suggests that customers who have been with the company for a long time may be more at risk of leaving. 3. payment_failures (0.544): Payment failures are an important indicator of customer churn. This suggests that issues related to payment processing, billing, or financial difficulties can lead to customer dissatisfaction and churn. Features with Moderate Importance: 1. monthly_logins (0.480): Monthly logins indicate customer engagement, and a lack of engagement (i.e., infrequent logins) can be a sign of churn. This suggests that customers who are not actively using the service may be more likely to leave. 2. tenure_fee_interaction (0.312): The way tenure intersects with fee-related issues (e.g., pricing, pricing changes) seems to be somewhat significant, however, the exact relationship between these is unclear. Features with Lower Importance: 1. country_Bangladesh (0.129): The country of residence has a relatively low importance in predicting customer churn. This suggests that geographic location may not be a significant factor in determining customer loyalty. 2. escalations (0.126): Escalations (i.e., issues escalated to higher-level support) are not a significant predictor of customer churn. This suggests that while escalations may indicate quality issues, they are not a primary driver of churn. 3. email_open_rate (0.111): Customer engagement metrics (e.g., email open rates) are relatively low in importance, indicating that they may not be a significant predictor of customer churn. 4. city_Berlin (0.093): The city of residence also has a relatively low importance in predicting customer churn. 5. city_London (0.067): Like other geographic features, the city of London has a low importance in predicting customer churn. Takeaways: 1. Focus on customer satisfaction and experience: The high importance of csat_score suggests that improving customer satisfaction through better support, communication, and quality is critical. 2. Pay attention to payment and billing: Payment failures and issues related to payment processing can lead to churn, so addressing these issues is essential. 3. Monitor engagement metrics: While not as critical, monthly logins and email open rates can still indicate customer engagement or disengagement, which may be useful in early identification of potential churn. 4. Consider tenure-based insights: Analyze customer tenure to identify potential churn risks, and consider ways to retain long-term customers (e.g., loyalty programs, special offers). 5. Rethink geographic factors: The relatively low importance of geographic features (e.g., country, city) suggests that they may not be significant drivers of customer churn."

**Overall Model Summary**     

Based on the metrics of Accuracy, Precision, Recall, F2 Score, AUC and Profit/Loss, the `Decision Tree Classifier` is the winner. `Random Forest Classifier` and `XGBoost Classifier` are virtually tied for the second spot (see comparison table below). 

For this particular dataset, the `Decision Tree Classifier` is the recommended machine learning algorithm.

| Model | Accuracy | Precision | Recall | F2 Score | AUC |Profit/Loss ($)|
|:---------|:---------|:---------|:---------|:---------|:---------|:--------|
| XGBoost | 0.7380 | 0.2659 | 0.9396 | 0.6124 | 0.840 | 43,620 |
| Random Forest | 0.7393 | 0.2569 | 0.9396 | 0.6135 | 0.842 | 43,700 |
| Decision Tree | 0.7469 | 0.2627 | 0.9396 | 0.6200 | 0.839 | 44,180 | 
| Keras | 0.6108 | 0.1861 | 0.9329 | 0.5175 | 0.820 | 34,620 | 
| Logistic Regression | 0.3042 | 0.1170 | 0.9799 | 0.3959 | 0.733 | 21,580 |
| SVC | 0.4754 | 0.1452 | 0.9396 | 0.4487 | 0.782 | 26,940 |
| K-Neighbors | 0.4641 | 0.1426 | 0.9396 | 0.4436 | 0.770 | 26,220 |

The **feature importance** is consistent with what is highlighted in the **Heatmap of Significant Correlations** in the following order: `csat_score`, `tenure_months`, `payment_failures`, and `monthly_logins`.

**Next Steps and Further Recommendations**  

- Confirm the model that will suit the business needs in terms of the optimal level of churn identification and precision.    
- Continue model development to include actual identification of clients who are likely to churn.
- Tune the thresholds to maximize profit using realistic lifetime value and cost of retention assumptions.
- Deploy and apply the model for the use of relevant business groups like marketing teams for retention campaigns, customer support teams for proactive outreach, and product teams for feature improvements. Leadership can also be guided by the model for churn strategy and forecasting.     
- Continue model development to validate the features relative importance to guide management on which features need to be given particular attention in order to prevent churn.
  
Churn in this dataset is primarily driven by customer satisfaction and engagement levels rather than pricing. Improving user experience and increasing product adoption would likely have the strongest impact on reducing churn.

**Notebook**    
You can view the full analysis here:

[Exploratory Data Analysis](01_EDA_Customer_Churn_Prediction.ipynb)
[Logistic_Regression](02_Logistic_Regression_Customer_Churn_Prediction.ipynb)
[Decision_Tree](03_Decision_Tree_Customer_Churn_Prediction.ipynb)
[KNearest_Neighbors](04_KNearest_Neighbors_Customer_Churn_Prediction.ipynb)
[Support_Vector](05_Support_Vector_Customer_Churn_Prediction.ipynb)
[Random_Forest](06_Random_Forest_Customer_Churn_Prediction.ipynb)
[Keras](07_Keras_Customer_Churn_Prediction.ipynb)
[XGBoost](08_XGBoost_Customer_Churn_Prediction.ipynb)
[Summary](09_Summary_Customer_Churn_Prediction.ipynb)

**References:** 
Ronaldo Bantayan (Author) Email: one01bant@yahoo.com     
in fulfillment of requirements: Professional Certificate in Machine Learning and Artificial Intelligence UC Berkely Engineering March 22, 2026

