###---------------Bank Loan Decision Making Analytics-------------------
# 3. Data Preprocessing (Attributes definition, data exploration, checking missing value, checking zero and more)
# Load necessary libraries
install.packages("VIM")
library(tidyverse)
library(caret)
library(corrplot)
library(ggplot2)
library(MASS)
library(glmnet)
library(randomForest)
library(VIM)
library(pROC)
library(rpart)
library(rpart.plot)

# Load the data
loan.data <- read.csv("C:/Users/13142/Desktop/M.S/Analytical Practicum/Case 2/loan.csv")

# Data exploration
dim(loan.data)
head(loan.data)
str(loan.data)
summary(loan.data)

# Checking missing values
missing.values <- colSums(is.na(loan.data))
print(missing.values)

# Visualize missing values using VIM
numeric_cols <- loan.data %>% select_if(is.numeric)
missing.plot.numeric <- numeric_cols %>%
  pivot_longer(cols = everything(), names_to = "key", values_to = "val") %>%
  mutate(is.missing = is.na(val)) %>%
  ggplot(aes(x = key, fill = is.missing)) +
  geom_bar() +
  scale_fill_manual(values = c("TRUE" = "red", "FALSE" = "green")) +
  theme_minimal()
print(missing.plot.numeric)
aggr.plot <- aggr(loan.data, col=c('navyblue','red'), numbers=TRUE, sortVars=TRUE, 
                  labels=names(loan.data), cex.axis=.7, gap=3, ylab=c("Missing data", "Pattern"))

# Remove rows with missing values
loan.data_clean <- na.omit(loan.data)
cat("\nRows before removing missing values:", nrow(loan.data), "\n")
cat("Rows after removing missing values:", nrow(loan.data_clean), "\n")

# Checking zero values in numeric columns
zero_values <- colSums(loan.data_clean == 0, na.rm = TRUE)
print(zero_values)

# Convert categorical variables to factors
loan.data_clean$REASON <- as.factor(loan.data_clean$REASON)
loan.data_clean$JOB <- as.factor(loan.data_clean$JOB)
loan.data_clean$BAD <- as.factor(loan.data_clean$BAD)

# 4. Predictor Analysis and Relevancy
# One-hot encoding of categorical variables
loan.data_clean_encoded <- loan.data_clean %>% mutate_if(is.factor, as.numeric)

# Correlation Matrix for numeric variables
cor_matrix_all <- cor(loan.data_clean_encoded)
corrplot(cor_matrix_all, method = "circle", type = "upper", tl.cex = 0.7)

# Chi-square test for categorical variables
cat("\nChi-square test results:\n")
print(chisq.test(loan.data_clean$REASON, loan.data_clean$BAD))
print(chisq.test(loan.data_clean$JOB, loan.data_clean$BAD))

# Visualizing categorical variables and their relationship with BAD
ggplot(loan.data_clean, aes(x = REASON, fill = BAD)) +
  geom_bar(position = "fill") +
  labs(title = "Loan Default by Reason", x = "Reason", y = "Proportion") +
  theme_minimal()

ggplot(loan.data_clean, aes(x = JOB, fill = BAD)) +
  geom_bar(position = "fill") +
  labs(title = "Loan Default by Job", x = "Job", y = "Proportion") +
  theme_minimal()

# Visualizing numeric variables using boxplots and density plots
ggplot(loan.data_clean, aes(x = BAD, y = LOAN)) +
  geom_boxplot() +
  labs(title = "Loan Amount by Default Status", x = "Default Status", y = "Loan Amount") +
  theme_minimal()

ggplot(loan.data_clean, aes(x = YOJ, fill = BAD)) +
  geom_density(alpha = 0.5) +
  labs(title = "Distribution of Years at Job by Default Status", x = "Years at Job", y = "Density") +
  theme_minimal()

# Scatter plot for LOAN vs VALUE, colored by BAD
ggplot(loan.data_clean, aes(x = LOAN, y = VALUE, color = BAD)) +
  geom_point(alpha = 0.7) +
  labs(title = "Loan Amount vs Home Value by Default Status", x = "Loan Amount", y = "Home Value") +
  theme_minimal()

# Facet bar plots for categorical predictors (REASON and JOB)
ggplot(loan.data_clean, aes(x = BAD, fill = BAD)) +
  geom_bar() +
  facet_wrap(~ REASON + JOB) +
  labs(title = "Faceted Loan Default by Reason and Job", x = "Default Status", y = "Count") +
  theme_minimal()

# Categorical predictors
categorical_vars <- names(loan.data_clean)[sapply(loan.data_clean, is.factor)]

# Numeric predictors
numeric_vars <- names(loan.data_clean)[sapply(loan.data_clean, is.numeric)]

# Visualize categorical predictors
for (var in categorical_vars) {
  p <- ggplot(loan.data_clean, aes_string(x = var, fill = "BAD")) +
    geom_bar(position = "fill") +
    labs(title = paste("Proportion of Loan Default by", var), x = var, y = "Proportion") +
    theme_minimal()
  print(p)
}

# Visualize numeric predictors
for (var in numeric_vars) {
  p <- ggplot(loan.data_clean, aes_string(x = "BAD", y = var)) +
    geom_boxplot() +
    labs(title = paste("Loan Default by", var), x = "Default Status", y = var) +
    theme_minimal()
  print(p)
}

# Build Random Forest Model
set.seed(321)
rf_model <- randomForest(BAD ~ ., data = loan.data_clean, importance = TRUE)

# Print Feature Importance Scores
importance_scores <- importance(rf_model)
print(importance_scores)
varImpPlot(rf_model, main = "Variable Importance Plot (Random Forest)")

# Data Transformation
# Normalize numeric variables
numeric_vars <- names(loan.data_clean)[sapply(loan.data_clean, is.numeric)]
loan.data_normalized <- loan.data_clean %>%
  mutate(across(all_of(numeric_vars), scale)) %>%
  mutate(BAD = factor(BAD, levels = c("0", "1")))

# Handle Imbalanced Data (Random Oversampling)
set.seed(123)
majority_class <- loan.data_normalized %>% filter(BAD == "0")
minority_class <- loan.data_normalized %>% filter(BAD == "1")

# Oversample the minority class to match the majority class size
minority_class_oversampled <- minority_class[sample(nrow(minority_class), nrow(majority_class), replace = TRUE), ]
loan.data_balanced <- rbind(majority_class, minority_class_oversampled)

# Data Partitioning Methods
# Verify class distribution
table(loan.data_balanced$BAD)

# Data Partitioning
set.seed(123)
train_index <- createDataPartition(loan.data_balanced$BAD, p = 0.7, list = FALSE)
train_data <- loan.data_balanced[train_index, ]
test_data <- loan.data_balanced[-train_index, ]
dim(train_data)
dim(test_data)
head(train_data)
head(test_data)

# Model Fitting, Validation Accuracy and Test Accuracy
# Logistic Regression Model
cat("\n=== Training Logistic Regression Model ===\n")
logistic_model <- glm(BAD ~ ., data = train_data, family = "binomial")
summary(logistic_model)

# Decision Tree Model
cat("\n=== Training Decision Tree Model ===\n")
tree_model <- rpart(BAD ~ ., data = train_data, method = "class")
printcp(tree_model)

# Visualize the decision tree using rpart.plot
rpart.plot(tree_model, type = 3, extra = 104, under = TRUE, fallen.leaves = TRUE, 
           main = "Decision Tree for Loan Default Prediction", tweak = 1.2)

# Report Model Performance
# Logistic Regression Performance
logistic_preds <- predict(logistic_model, newdata = test_data, type = "response")
logistic_class_preds <- ifelse(logistic_preds > 0.5, 1, 0)
logistic_conf_matrix <- confusionMatrix(as.factor(logistic_class_preds), test_data$BAD, positive = "1")
print(logistic_conf_matrix)

# Accuracy
logistic_accuracy <- logistic_conf_matrix$overall["Accuracy"]

# ROC and AUC for Logistic Regression
logistic_roc <- roc(test_data$BAD, logistic_preds)
cat("Logistic Regression AUC: ", auc(logistic_roc), "\n")
plot(logistic_roc, main = "ROC Curve - Logistic Regression")

# Decision Tree Performance
tree_preds <- predict(tree_model, newdata = test_data, type = "class")
tree_conf_matrix <- confusionMatrix(tree_preds, test_data$BAD, positive = "1")
print(tree_conf_matrix)

# Accuracy
tree_accuracy <- tree_conf_matrix$overall["Accuracy"]

# ROC and AUC for Decision Tree
tree_roc <- roc(test_data$BAD, as.numeric(tree_preds))
cat("Decision Tree AUC: ", auc(tree_roc), "\n")
plot(tree_roc, main = "ROC Curve - Decision Tree")

plot(logistic_roc, col = "blue", lwd = 2, main = "ROC Curves: Logistic Regression vs Decision Tree")
plot(tree_roc, col = "red", lwd = 2, add = TRUE)
abline(a = 0, b = 1, lty = 2, col = "gray")
legend("bottomright", legend = c("Logistic Regression", "Decision Tree"),
       col = c("blue", "red"), lwd = 2)

# Logistic Regression Performance
logistic_preds <- predict(logistic_model, newdata = test_data, type = "response")
logistic_class_preds <- ifelse(logistic_preds > 0.5, 1, 0)
logistic_conf_matrix <- confusionMatrix(as.factor(logistic_class_preds), test_data$BAD, positive = "1")
print(logistic_conf_matrix)

# Accuracy
logistic_accuracy <- logistic_conf_matrix$overall["Accuracy"]

# Calculate Cost Matrix for Logistic Regression
TP <- logistic_conf_matrix$table[2, 2]  # True Positives
TN <- logistic_conf_matrix$table[1, 1]  # True Negatives
FP <- logistic_conf_matrix$table[1, 2]  # False Positives
FN <- logistic_conf_matrix$table[2, 1]  # False Negatives

# Define the costs
cost_fp <- 10000  # Cost for False Positive
cost_fn <- 5000   # Cost for False Negative

# Calculate total cost for logistic regression
total_cost_logistic <- (FP * cost_fp) + (FN * cost_fn)

# Print the cost matrix and total cost
cat("Logistic Regression - Cost Matrix:\n")
cost_matrix_logistic <- matrix(c(0, cost_fn, cost_fp, 0), nrow = 2, byrow = TRUE,
                               dimnames = list(c("Actual 1", "Actual 0"), c("Predicted 1", "Predicted 0")))
print(cost_matrix_logistic)
cat("\nTotal Cost of Misclassification (Logistic Regression):", total_cost_logistic, "\n")

# ROC and AUC for Logistic Regression
logistic_roc <- roc(test_data$BAD, logistic_preds)
cat("Logistic Regression AUC: ", auc(logistic_roc), "\n")
plot(logistic_roc, main = "ROC Curve - Logistic Regression")

# Decision Tree Performance
tree_preds <- predict(tree_model, newdata = test_data, type = "class")
tree_conf_matrix <- confusionMatrix(tree_preds, test_data$BAD, positive = "1")
print(tree_conf_matrix)

# Accuracy
tree_accuracy <- tree_conf_matrix$overall["Accuracy"]

# Calculate Cost Matrix for Decision Tree
TP_tree <- tree_conf_matrix$table[2, 2]  # True Positives
TN_tree <- tree_conf_matrix$table[1, 1]  # True Negatives
FP_tree <- tree_conf_matrix$table[1, 2]  # False Positives
FN_tree <- tree_conf_matrix$table[2, 1]  # False Negatives

# Calculate total cost for decision tree
total_cost_tree <- (FP_tree * cost_fp) + (FN_tree * cost_fn)

# Print the cost matrix and total cost
cat("Decision Tree - Cost Matrix:\n")
cost_matrix_tree <- matrix(c(0, cost_fn, cost_fp, 0), nrow = 2, byrow = TRUE,
                           dimnames = list(c("Actual 1", "Actual 0"), c("Predicted 1", "Predicted 0")))
print(cost_matrix_tree)
cat("\nTotal Cost of Misclassification (Decision Tree):", total_cost_tree, "\n")

# ROC and AUC for Decision Tree
tree_roc <- roc(test_data$BAD, as.numeric(tree_preds))
cat("Decision Tree AUC: ", auc(tree_roc), "\n")
plot(tree_roc, main = "ROC Curve - Decision Tree")

# Comparing ROC curves
plot(logistic_roc, col = "blue", lwd = 2, main = "ROC Curves: Logistic Regression vs Decision Tree")
plot(tree_roc, col = "red", lwd = 2, add = TRUE)
abline(a = 0, b = 1, lty = 2, col = "gray")
legend("bottomright", legend = c("Logistic Regression", "Decision Tree"),
       col = c("blue", "red"), lwd = 2)
