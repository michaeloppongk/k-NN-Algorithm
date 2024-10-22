# Load necessary libraries
library(readr)  # For reading CSV files
library(caret)  # For data preprocessing and modeling

# Load the dataset
my_data_bank <- read.csv("UniversalBank.csv")

# Convert 'PersonalLoan' column to a factor since it's a categorical variable
my_data_bank$PersonalLoan <- as.factor(my_data_bank$PersonalLoan)

# Create dummy variables for the 'Education' column
my_data_bank$Education_1 <- ifelse(my_data_bank$Education == 1, 1, 0)
my_data_bank$Education_2 <- ifelse(my_data_bank$Education == 2, 1, 0)

# Standardize the numeric predictors (Age, Experience, Income, Family)
# Scaling the values between 0 and 1
preprocess_model <- preProcess(my_data_bank[, c("Age", "Experience", "Income", "Family")],
                               method = "range")

my_data <- predict(preprocess_model, my_data_bank)

# Model 1: KNN model with k = 3
model_1 <- knn3(PersonalLoan ~ Age + Experience + Income + Family + Education_1 + Education_2,
                data = my_data,
                k = 3)
print(model_1)

# Make predictions on the training data
my_data$PersonalLoan_hat <- predict(model_1, newdata = my_data, type = "class")

# Generate confusion matrix for model evaluation
confusionMatrix(my_data$PersonalLoan, my_data$PersonalLoan_hat)

# Model 2: Tune the KNN model with cross-validation and select the best 'k' value
set.seed(1234)
model_2 <- train(PersonalLoan ~ Age + Experience + Income + Family + Education_1 + Education_2,
                 data = my_data,
                 method = "knn",
                 trControl = trainControl(method = "cv", number = 10),  # 10-fold cross-validation
                 tuneGrid = data.frame(k = c(1, 3, 5, 7, 9, 11, 13, 15, 17)))  # Tune for multiple k values

# Print and plot the results for model 2
print(model_2)
plot(model_2)

# Model 3: Extended tuning with tuneLength = 50 (automatically searches for the best 'k' from a wider range)
model_3 <- train(PersonalLoan ~ Age + Experience + Income + Family + Education_1 + Education_2,
                 data = my_data,
                 method = "knn",
                 trControl = trainControl(method = "cv", number = 10),  # 10-fold cross-validation
                 tuneLength = 50)  # Automatically select best 'k' from a larger range

# Print and plot the results for model 3
print(model_3)
plot(model_3)

# Select the best model from Model 2 (k = 5)
best_model <- knn3(PersonalLoan ~ Age + Experience + Income + Family + Education_1 + Education_2,
                   data = my_data,
                   k = 5)

# Make predictions using the best model
my_data$PersonalLoan_hat_2 <- predict(best_model, newdata = my_data, type = "class")

# Generate confusion matrix for the best model
confusionMatrix(my_data$PersonalLoan, my_data$PersonalLoan_hat_2)
