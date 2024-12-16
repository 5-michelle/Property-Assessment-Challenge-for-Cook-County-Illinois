## About 20 min run. 

# Load libraries
library(tidyverse)
library(caret)

# Load data
history_data <- read_csv("historic_property_data.csv")

# Preprocessing ############################################################################################################################

# Separate the target variable
target <- history_data$sale_price
history_data <- history_data %>% select(-sale_price)

## Split columns into numeric and categorical
numeric_cols <- sapply(history_data, is.numeric)
categorical_cols <- !numeric_cols

### Numeric: Replace missing values with the median
params_numeric <- preProcess(history_data[, numeric_cols], method = c("medianImpute"))
history_data[, numeric_cols] <- predict(params_numeric, history_data[, numeric_cols])

### Categorical: Replace missing values with "unknown"
history_data[, categorical_cols] <- lapply(history_data[, categorical_cols], function(col) {
  ifelse(is.na(col), "unknown", col)
})

### Scale numeric variables
scaling_params <- preProcess(history_data[, numeric_cols], method = c("center", "scale"))
history_data[, numeric_cols] <- predict(scaling_params, history_data[, numeric_cols])

# Add the target variable back
history_data$sale_price <- target

# Model ####################################################################################################################################
## Set seed and training data rate
set.seed(123)
train_index <- sample(1:nrow(history_data), 0.8 * nrow(history_data))

## Random Forest #####################################################
library(randomForest)
rf_train_data <- history_data[train_index, ]
rf_test_data <- history_data[-train_index, ]

### Set mtry times
mtry_value <- floor(sqrt(ncol(rf_train_data) - 1)) 

### Train model
rf_model <- randomForest(
  sale_price ~ ., 
  data = rf_train_data, 
  importance = TRUE, 
  ntree = 500, 
  mtry = mtry_value
)

### Make predictions
rf_predictions <- predict(rf_model, rf_test_data)
rf_mse <- mean((rf_test_data$sale_price - rf_predictions)^2)
print(paste("Random Forest MSE:", rf_mse))

## XGBoost ###########################################################
library(xgboost)

# Prepare xgb_data
xgb_data <- history_data
xgb_numeric_cols <- sapply(history_data, is.numeric)
xgb_categorical_cols <- !xgb_numeric_cols

# Convert categorical columns to numeric 
xgb_data[, xgb_categorical_cols] <- lapply(xgb_data[, xgb_categorical_cols], as.factor)
xgb_data[, xgb_categorical_cols] <- lapply(xgb_data[, xgb_categorical_cols], function(col) as.numeric(as.factor(col)))

# Split data into training and testing sets
xgb_train_data <- xgb_data[train_index, ]
xgb_test_data <- xgb_data[-train_index, ]

# Separate features and target variable
train_x <- xgb_train_data %>% select(-sale_price) 
train_y <- xgb_train_data$sale_price             
test_x <- xgb_test_data %>% select(-sale_price)  
test_y <- xgb_test_data$sale_price               

# Ensure train_x and test_x are numeric matrices
train_x <- as.matrix(train_x)
test_x <- as.matrix(test_x)

# Create DMatrix for XGBoost
dtrain <- xgb.DMatrix(data = train_x, label = train_y)
dtest <- xgb.DMatrix(data = test_x, label = test_y)


# Set parameters for XGBoost
params <- list(
  booster = "gbtree",
  eta = 0.01,                      
  max_depth = 6,                  
  subsample = 0.8,                
  colsample_bytree = 0.8,
  base_score = mean(history_data$sale_price)
)

# Custom objective function to penalize negative predictions
custom_objective <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  residuals <- preds - labels
  grad <- ifelse(preds < 0, 2 * residuals - 30, 2 * residuals)  # Penalize negatives
  hess <- ifelse(preds < 0, 2 + 5, 2)  # Adjust hessian for penalization
  list(grad = grad, hess = hess)
}

# Train model
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 5000,       
  watchlist = list(train = dtrain, test = dtest),
  print_every_n = 100,
  early_stopping_rounds = 25, 
  obj = custom_objective,
  maximize = FALSE
)

### Make predictions 
xgb_predictions <- predict(xgb_model, dtest)

# Calculate MSE
xgb_mse <- mean((xgb_test_data$sale_price - xgb_predictions)^2)
print(paste("XGBoost MSE:", xgb_mse))

# Result: Since the random forest model takes more time to process and XGBoost performs better on this data, we have chosen to use XGBoost as our final model.

#### Load predict_data #####################################################################################################################
predict_data <- read_csv("predict_property_data.csv")

# Identify numeric and categorical columns
numeric_cols_pred <- sapply(predict_data, is.numeric)
categorical_cols_pred <- !numeric_cols_pred

# Apply the same numeric preprocessing
predict_data[, numeric_cols_pred] <- predict(params_numeric, predict_data[, numeric_cols_pred])
predict_data[, numeric_cols_pred] <- predict(scaling_params, predict_data[, numeric_cols_pred])

# Apply the same categorical preprocessing
predict_data[, categorical_cols_pred] <- lapply(predict_data[, categorical_cols_pred], function(col) {
  ifelse(is.na(col), "unknown", col)
})
predict_data[, categorical_cols_pred] <- lapply(predict_data[, categorical_cols_pred], as.factor)
predict_data[, categorical_cols_pred] <- lapply(predict_data[, categorical_cols_pred], function(col) as.numeric(as.factor(col)))

# Convert `predict_data` to DMatrix
predict_data <- predict_data[, -which(names(predict_data) == "pid")]
dpredict <- xgb.DMatrix(data = as.matrix(predict_data))

# Generate predictions
predictions <- predict(xgb_model, newdata = dpredict)

# Combine predictions with 'pid'
output <- data.frame(
  pid = seq(1, nrow(predict_data)),
  assessed_value = predictions
)
summary(output)

# Export the results to a CSV file
write_csv(output, "assessed_value.csv")