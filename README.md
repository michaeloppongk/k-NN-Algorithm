**Analysis Report: Predicting Personal Loan Approval using k-Nearest Neighbors**
**Objective**
This analysis aims to predict whether a customer will accept a personal loan using the k-Nearest Neighbors (k-NN) algorithm, focusing on factors like age, experience, income, family size, and education levels.

**Data Preprocessing**
The dataset used contains various customer attributes including:

Age
Experience
Income
Family Size
Education Level (converted into binary indicators)
The target variable, PersonalLoan, is a categorical variable indicating whether a customer accepted a loan (1) or not (0). This column was converted to a factor type to reflect its categorical nature.

The education levels were divided into binary columns:

Education_1: Customers with education level 1.
Education_2: Customers with education level 2.
A normalization technique was applied to scale features (Age, Experience, Income, Family Size) to a range of 0 to 1 using the preProcess function from the caret package.

**Modeling Approaches**
Model 1: k-NN with k = 3
The initial k-NN model was built with k = 3 to predict whether a customer will accept a loan or not. The model used six predictors: Age, Experience, Income, Family Size, Education_1, and Education_2.

**Training Set Performance**:

Accuracy: 98.72%
Sensitivity (True Positive Rate): 98.67%
Specificity (True Negative Rate): 99.29%
Kappa: 0.922 (indicates high agreement between predicted and actual classifications)
Observations:

The model performed well with an accuracy of 98.72%, indicating a high ability to correctly classify personal loan acceptance.
However, there was a slight misclassification in predicting PersonalLoan = 1, with 61 false negatives and 3 false positives.
Model 2: Cross-Validated k-NN with Multiple k Values
To improve the performance, the model was tuned using cross-validation (10-fold) to determine the best k value. Several values of k (from 1 to 17) were evaluated.

**Key Results**:

Best k: The best model was selected with k = 3, achieving an accuracy of 97.76%.
Model Performance for different k-values revealed that smaller values of k (1, 3) had higher accuracy, while larger k values saw a decline in accuracy and Kappa.
Final Accuracy for k = 3: 97.76%

Model 3: Extended Tuning with Larger k Values (tuneLength = 50)
An extended range of k-values (1 to 103) was tested to ensure the optimal value for the model. The best result was achieved with k = 5, leading to a slight improvement over previous models.

**Final Model (k = 5)**:

Accuracy: 98.28%
Sensitivity: 98.24%
Specificity: 98.76%
Kappa: 0.8934
Confusion Matrix for the Best Model (k = 5):

True Positives (Correctly predicted PersonalLoan = 0): 4515
True Negatives (Correctly predicted PersonalLoan = 1): 399
False Positives: 5
False Negatives: 81
Model Comparison
Model 1 (k = 3) had the highest accuracy (98.72%) and a Kappa value of 0.922, indicating excellent agreement. However, it had more false negatives (61).
Model 2 explored various k-values and confirmed that k = 3 was the best option, but with slightly lower accuracy (97.76%).
Model 3 (k = 5), selected after extended tuning, provided a balanced trade-off with an accuracy of 98.28% and a lower number of false positives (5), but more false negatives (81).

**Conclusion**
The k-NN model with k = 3 performed the best in terms of overall accuracy and Kappa, indicating strong predictive power for personal loan acceptance. However, the extended tuning model (k = 5) provided a marginally more balanced prediction in terms of reducing false positives, but at the cost of more false negatives.

**Recommendation**: The k = 3 model is recommended as it provides the highest accuracy and balanced predictions between false positives and negatives. Future work can explore further feature engineering or more advanced algorithms like logistic regression or decision trees for potentially better performance.
