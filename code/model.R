##################################################################################################
# File: model.R
# Author: Sam Seatt
# Creation Date: May 15, 2019
#
# HarvardX/edX Data Science Capstone - CYO - Modeling and Predicting Heart Disease using the
# Kaggle Framingham Heart Study dataset
#
# File Path: code/model.R
# Project Name: fram-heart
#
# Project GitHub Repository:
#    https://github.com/samseatt/fram-heart
#
# Framingham Heart Study dataset location:
#    https://www.kaggle.com/amanajmera1/framingham-heart-study-dataset
#    https://raw.githubusercontent.com/samseatt/fram-heart/master/data/framingham.csv
# 
# UCI Heart Disease dataset location:
#    https://www.kaggle.com/ronitf/heart-disease-uci
#    https://raw.githubusercontent.com/samseatt/fram-heart/master/data/uci.csv
#
##################################################################################################
# The purpose of this R script file is to provides various methods to model, train and capture the
# results of various competing models to analyze this binary classificaton problem
#
# This file has two parts, each using a differnt heart disease data set from Kaggle:
#
#  - The first part uses the Framingham Heart Study dataset from Kaggle. This dataset is larger
#    and lest structured compared to the second dataset. The inputs are also less specific, so
#    code is provided to extensively explore, sanitize, analyze, and consolidate this data.
#    The rest of the code of this part involves partitioning the data for traiing, training
#    different models, and saving the results for analysis in the RMD file that generates the
#    project report.
#
# - The second part uses the Heart Disease UCI dataset. This is a cleaner dataset, but contains
#   very few rows and presents with an undertraining challenge. The same models are then
#   re-written to run under this dataset. The results are then made available for use in the
#   RMD report.
#     
# The code is meant to assist in analysis of various binary classification models and pick the
# best model for the dataset and its intended produce (prediction of heart disease risk).
# A secondary purpose is to evaluate the performance of the two differnet data sets and
# understand their relative impacts on our model; this latter effort is to understand the
# disease correlates (independent variabls that contribute towards prediction) in order to
# design the most appropriate data and the right data collection campaigns.

# This dataset and the methods and results of the R code in this file are meant to be used as part
# of the CYO (Choose Your Own) project for the HarvardX Data Science Capstone project for
# Sam Seatt (edX id: ).
#
# This script (model.R) is also available in my RStudio project checked in GitHub at a location listed
# in this file's header.
##################################################################################################

##################################################################################################
# Load the necessary libraries that I will use throught this analysis
##################################################################################################
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
library(dplyr)
library(ggplot2)
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

#####################################################################################
# Get the data from the Kaggle repository (indirectly via GitHub)
#####################################################################################
# Since it has proven difficult to download data in R directly from Kaggle, we will download it
# from my GitHub repository where I have already committed the my data file that I manually downloaded
# from Kaggle (Note: this data file can also be directly copied from my GitHub repository from
# data/heart.csv in my project)
library(readr)  # for read_csv
library(knitr)  # for kable
myfile <- "https://raw.githubusercontent.com/samseatt/fram-heart/master/data/framingham.csv"

# Save the original file
fram_heart <- read_csv(myfile)

# Make a copy for further data cleaning and prepping the dataset for modeling purposes
heart <- fram_heart

##################################################################################################
# Function: Print histogram segerated by male/female and healthy/deseased
##################################################################################################
plotOutcome <- function(df, dfcol, bw) {
  df %>%
  ggplot(aes(dfcol, fill = TenYearCHD)) + 
  geom_histogram(binwidth = bw, color = "black") +
  scale_x_continuous() + 
  facet_grid(male ~ TenYearCHD, labeller = labeller(male = label_both, TenYearCHD = label_both)) +
  geom_vline(xintercept = median(dfcol), color = "red")
}

##################################################################################################
# Preliminary Data Analysis
# -------------------------
# 
# Study some basic properties of the data and conduct some preliminary analysis before moving on
# to full-fledged data exploration.
#
# Such preliminary step will aid in data cleaning and sanitization requirements e.g., looking at
# missing / NA data values etc. Such analysis will also help determine the overall training strength
# of the data e.g. total number or rows available for training, variability of date for each input,
# whether or not any inputs need to be normalized, which columns are more useful and which columnes
# are of no or little use and thus could be dropped, etc.
#
# (Once this priliminary analysis is performed, it will help me clean the data and particion it.
# After which, I can proceed with other data analysis task that would instead be more useful for
# the purposes of modeling, training, and interpretation of the results)
##################################################################################################

##################################################################################################
# Understanding the data
##################################################################################################
class(heart)
head(heart)
str(heart)
# This tells me that I have read the CSV file as a dataframe as expected. There are 16 columns.
# 15 of these columns are various parameters or measurements form each patient, and one column:
# TenYearCHD (whether or not the patient developed Coronary Heart Disease within 10 years after
# the study i.e. ten years after taking the measurement) is the output or the label i.e. what
# my models need to match in training and predict
#.
# These details of the data are furhter summarized in the final report (RMD/PDF file).
# 
# It also tells me that all the columns are numeric - which is what's required for inputs of a
# machine learning model.
#
# Furthermore, the inputs have more or less within two orders of magnitude (between single digit to hundreds)
# so normalization of the data may not be necessary. Although some values are binary, but they
# are represented by 0 or 1 which may be a little smaller compared to total cholesterol (toChol)
# numbers that go into 300s. We can check the range of these to see if it will help to subtract
# the mean or the lowest value form each toChol reading. The same can be considered for sysBP, diaBP,
# BMI, heartRate and glucose.

##################################################################################################
# Exploring data relationships
# ----------------------------
# Check correlations beteen inputs (independenat variables that ideally should not be correlated in order to
# get the maximum benfit of each) and beteen imputs and output (the dependant variable, where correlation should exist)


####################################################
# See the summary of our data frame
####################################################
summary(heart)
# This gives us the ranges and means of each parameter (and the output). In the case of total cholesterol,
# for example I can subract the mean from each value and then divide each value by 10 to get the
# data closer to 1. This can be helpful in keeping the relative influence of each input relatively
# even keel i.e. similar. On the other hand I will lose the ability to visualize my data properly
# (e.g when age appears between -1 and 1, with say -1 for 32 years, 0 for 49 years, and 1 for 70 years,
# interpreting itermediate results witht un-conversion will be tedious). Since the magnitudes don't
# seem to be several orders different, I will train on these outputs as-is.


##################################################################################################
# Data cleaning and sanitization
#
# I will clean my data before I partition the data into training and test data sets, so I don't
# have to do it twice on each set
##################################################################################################
# Rename output column to y
names(heart)[names(heart) == 'TenYearCHD'] <- 'y'

####################################################
# Factorize the binary classification output
####################################################
# Since I am working on a classification problem, I need to change target to factors i.e.
# as discrete labels representing diseased or not diseased (1 or 0), respectively and not
# magnitudes of 0 and 1. This is preferred for working with classification models.
heart$y <- as.factor(heart$y)

####################################################
# Remove column with too many NA values (for two reasons - elaborated a few lines below)
####################################################

# In my data wrangling exercise I do see several NA values (645 in total). I would need to remove these as part of
# data cleaning
nrow(heart)
sum(is.na.data.frame(heart))
# WIth 388 in glucose
colSums(is.na.data.frame(heart))

# This will be a significant (but not overwhelming) loss of 9.2% of the data. So let's first see how
# useful glucose level is in predicting heart disease.
heart %>%
  ggplot(aes(glucose, fill = y)) + 
  geom_histogram(binwidth = 20, color = "black") +
  scale_x_continuous() + 
  facet_grid(male ~ y, labeller = labeller(male = label_both, y = label_both)) +
  geom_vline(xintercept = median(heart$glucose, na.rm=TRUE), color = "red")
heart %>% group_by(male, y) %>% summarise(n = n())

# Well, it seems fairly even key for both healthy and diseased outcomes, but healthy subjects slightly
# favor towards lower glucose levels (bigger red bars on the left side, for both genders)
# The contribution is very slight, and therefore, I decide to save the row and simply drop the glucose
# column.
# Another secondary reason for removing this columns with high NA is that it will also be missing more
# but in real life. This would be a good time to ask the subject matter experts to see why the blood glucose
# information is not readily available in patient data (a reason would be that it is considered more related
# to diabetes or diabetes mediate heart disease, and may not be always ordered in the blood test. So it is
# not likely that much tied to heart disease even from the point of view of the medical professionals)

# First save the the original version with glucose, in case we need to compare its relative influence
# of any given algorithm. (In order to not further increase the scope of this project, I will do these
# evaluation outside this project, and at a later time.)
heart_with_glucose <- na.omit(heart)

# Remove glucose column
heart$glucose <- NULL

####################################################
# Remove possible Human Bias from the dataset
####################################################
# I also drop education as it is medically non-relevant (though it could potentially be indirectly correlated to
# nutritional habits). I it appears that this field has the potential of introducing training and social bias
# in the ML models (as they will be later used for treatment options during the prediction phase).
# Also, below visualization helps us infer that this variable trends the same for both healthy and at-risk subjects.
heart %>%
  ggplot(aes(education, fill = y)) + 
  geom_histogram(binwidth = 1, color = "black") +
  scale_x_continuous() + 
  facet_grid(male ~ y, labeller = labeller(male = label_both, y = label_both)) +
  geom_vline(xintercept = median(heart$education), color = "red")

# Remove education column
heart$education <- NULL

####################################################
# Get rid of the remaining NAs
####################################################
# Now instead of dropping further columns, I drop all rows with one or more NA column values
heart <- na.omit(heart)
nrow(heart)
# Now I have 4090 rows to train and validate. Seems like just about right to get a good representative
# training in reasonable time, allowing to run the models relatively quickly and comparing multipe models
# (With a very large dataset, even though I can train better, albeit risking overtraiing if I'm not
# careful, it will take a long time to crunch throught the traiing, therefore limiting me from trying
# and comparing several model).

# Let's see if we have sufficient positive preditions.
sum(as.numeric(as.character(heart$y)))
nrow(heart) - sum(as.numeric(as.character(heart$y)))
# I see 611 positive outcomes for 3479 negative, to work with, and that should be sufficient

# So in summary, this data should work quite well for my project.

##################################################################################################
# Functional analysis and further data exploration
##################################################################################################
hist(heart$age)
hist(heart$totChol)

# See the distribution of totChol between diseased and healthy patinets, furhter divided by sex
heart %>% group_by(male, y) %>% summarise(n = n())

####################################################
# Relation (if any) between selected inputs
####################################################
#
# Correlaton between age and systolic blood pressure
ggplot(data = heart) + geom_point(aes(age, sysBP))
cor(heart$age, heart$sysBP)

# Correlation between age and cigarettes per day
ggplot(data = heart) + geom_point(aes(age, cigsPerDay))
cor(heart$age, heart$cigsPerDay)

# Correlation between age and total cholesterol
ggplot(data = heart) + geom_point(aes(age, totChol))
cor(heart$age, heart$totChol)

ggplot(data = heart) + geom_point(aes(sysBP, diaBP))
cor(heart$sysBP, heart$diaBP)
# As seen from t graph and the correlation values, the two blood pressures are correlated,
# however the correlation is complete, so I will keep both these inputs for the analysis as
# the second input can provide additional training information.


####################################################
# Relation between inputs and outputs (how each predicts heart disease risk)
####################################################
ggplot(data = fram_heart) + geom_boxplot(aes(as.factor(TenYearCHD), age), outlier.colour = "red", outlier.shape = 1, na.rm = TRUE) +
  labs(title = "Relative variability of Age", x = "Ten Year CHD", y = "Age")
ggplot(data = fram_heart) + geom_boxplot(aes(as.factor(TenYearCHD), totChol), outlier.colour = "red", outlier.shape = 1, na.rm = TRUE) +
  labs(title = "Relative variability of Total Cholesterol", x = "Ten Year CHD", y = "Total Cholesterol")
ggplot(data = fram_heart) + geom_boxplot(aes(as.factor(TenYearCHD), cigsPerDay), outlier.colour = "red", outlier.shape = 1, na.rm = TRUE) +
  labs(title = "Relative variability of Cigarettes Per Day", x = "Ten Year CHD", y = "Cigarettes Per Day")
# Average is close to zero cigarettes for health vs. higher for smokers

ggplot(data = fram_heart) + geom_boxplot(aes(as.factor(TenYearCHD), sysBP), outlier.colour = "red", outlier.shape = 1, na.rm = TRUE) +
  labs(title = "Relative variability of Systolic BP", x = "Ten Year CHD", y = "Systolic BP")
ggplot(data = fram_heart) + geom_boxplot(aes(as.factor(TenYearCHD), diaBP), outlier.colour = "red", outlier.shape = 1, na.rm = TRUE) +
  labs(title = "Relative variability of Diastolic BP", x = "Ten Year CHD", y = "Diastolic BP")
ggplot(data = fram_heart) + geom_boxplot(aes(as.factor(TenYearCHD), BMI), outlier.colour = "red", outlier.shape = 1, na.rm = TRUE) +
  labs(title = "Relative variability of Body Mass Index", x = "Ten Year CHD", y = "BMI")

ggplot(data = fram_heart) + geom_boxplot(aes(as.factor(TenYearCHD), glucose), outlier.colour = "red", outlier.shape = 1, na.rm = TRUE) +
  labs(title = "Relative variability of Glucose Level", x = "Ten Year CHD", y = "Blood Glucose")
# Heart Rate does not seem to be a good indicator

####################################################
# Plot histograms to see the role each input plays
####################################################
# Plot the histogram by each variable (further segregated into male/female) to see its relative
# impact on the 10-year onset chronic heart disease for each gender

# age
heart %>%
  ggplot(aes(age, fill = y)) + 
  geom_histogram(binwidth = 5, color = "black") +
  scale_x_continuous() + 
  facet_grid(male ~ y, labeller = labeller(male = label_both, y = label_both)) +
  geom_vline(xintercept = median(heart$age), color = "red")

# currentSmoker
heart %>%
  ggplot(aes(currentSmoker, fill = y)) + 
  geom_histogram(binwidth = 1, color = "black") +
  scale_x_continuous() + 
  facet_grid(male ~ y, labeller = labeller(male = label_both, y = label_both)) +
  geom_vline(xintercept = median(heart$currentSmoker), color = "red")

# cigsPerDay
heart %>%
  ggplot(aes(cigsPerDay, fill = y)) + 
  geom_histogram(binwidth = 5, color = "black") +
  scale_x_continuous() + 
  facet_grid(male ~ y, labeller = labeller(male = label_both, y = label_both)) +
  geom_vline(xintercept = median(heart$cigsPerDay), color = "red")

# BPMeds
heart %>%
  ggplot(aes(BPMeds, fill = y)) + 
  geom_histogram(binwidth = 1, color = "black") +
  scale_x_continuous() + 
  facet_grid(male ~ y, labeller = labeller(male = label_both, y = label_both)) +
  geom_vline(xintercept = median(heart$BPMeds), color = "red")

# prevalentStroke
heart %>%
  ggplot(aes(prevalentStroke, fill = y)) + 
  geom_histogram(binwidth = 1, color = "black") +
  scale_x_continuous() + 
  facet_grid(male ~ y, labeller = labeller(male = label_both, y = label_both)) +
  geom_vline(xintercept = median(heart$prevalentStroke), color = "red")

# prevalentHyp
heart %>%
  ggplot(aes(prevalentHyp, fill = y)) + 
  geom_histogram(binwidth = 1, color = "black") +
  scale_x_continuous() + 
  facet_grid(male ~ y, labeller = labeller(male = label_both, y = label_both)) +
  geom_vline(xintercept = median(heart$prevalentHyp), color = "red")

# diabetes
heart %>%
  ggplot(aes(diabetes, fill = y)) + 
  geom_histogram(binwidth = 1, color = "black") +
  scale_x_continuous() + 
  facet_grid(male ~ y, labeller = labeller(male = label_both, y = label_both)) +
  geom_vline(xintercept = median(heart$diabetes), color = "red")

# totChol
heart %>%
  ggplot(aes(totChol, fill = y)) + 
  geom_histogram(binwidth = 20, color = "black") +
  scale_x_continuous() + 
  facet_grid(male ~ y, labeller = labeller(male = label_both, y = label_both)) +
  geom_vline(xintercept = median(heart$totChol), color = "red")
summary(heart$totChol)

# sysBP
heart %>%
  ggplot(aes(sysBP, fill = y)) + 
  geom_histogram(binwidth = 20, color = "black") +
  scale_x_continuous() + 
  facet_grid(male ~ y, labeller = labeller(male = label_both, y = label_both)) +
  geom_vline(xintercept = median(heart$sysBP), color = "red")
summary(heart$sysBP)

# diaBP
heart %>%
  ggplot(aes(diaBP, fill = y)) + 
  geom_histogram(binwidth = 10, color = "black") +
  scale_x_continuous() + 
  facet_grid(male ~ y, labeller = labeller(male = label_both, y = label_both)) +
  geom_vline(xintercept = median(heart$diaBP), color = "red")
summary(heart$diaBP)

# BMI
heart %>%
  ggplot(aes(BMI, fill = y)) + 
  geom_histogram(binwidth = 5, color = "black") +
  scale_x_continuous() + 
  facet_grid(male ~ y, labeller = labeller(male = label_both, y = label_both)) +
  geom_vline(xintercept = median(heart$BMI), color = "red")
summary(heart$BMI)

# heartRate
heart %>%
  ggplot(aes(heartRate, fill = y)) + 
  geom_histogram(binwidth = 10, color = "black") +
  scale_x_continuous() + 
  facet_grid(male ~ y, labeller = labeller(male = label_both, y = label_both)) +
  geom_vline(xintercept = median(heart$heartRate), color = "red")
summary(heart$heartRate)

# Similarly there are several binary (0 or 1) values that ore logically factors, but they should also
# work well in the algorithms as their original numeric values of 0 and 1

# Also, prevelant stroke does not seem to be present much (only 21 times, and only slightly favors
# a diseased outcome). But because it has a slight tilt towards increased risk, I decide to just keep
# this input dangling there for now
heart %>% group_by(prevalentStroke, y) %>% summarise(n = n())


#####################################################################################
# Data Partitioning
# -----------------
# First, use the caret package to Split the data into training and test sets.
#
# Since I have significant amout of data (about 4000 rows), I allocate 20% of data
# for test and 8% for training, still giving me sufficient data to train
#####################################################################################
set.seed(1)
v_index <- createDataPartition(y = heart$y, times = 1, p = 0.1, list = FALSE)
other <- heart[-v_index,]
test <- heart[v_index,]

# ALso get a validation sample from the training data. The test data will be used for tuning
# parameters like determining the K for KNN. I am keeping it independent of the test data
# so I don't risk the integrity of the overall performance.
#
# (I will later use this data just to find the right K for KNN, so bootstrapping it out of the
# training data should not cause too much bias towards the training set)
t_index <- createDataPartition(y = other$y, times = 1, p = 0.1, list = FALSE)
val <- other[t_index,]
train <- other[-t_index,]

#####################################################################################
# Quick Check
#####################################################################################
# I have similar distribution of healthy and diseased patients in both training and test sets
# so the partition is done properly by caret as one would expect
mean(as.numeric(as.character(train$y)))
mean(as.numeric(as.character(test$y)))
mean(as.numeric(as.character(val$y)))

# # Try predicting based on cholesterol only
# glm_fit <- train %>% 
#   glm(y ~ totChol, data=., family = "binomial")
# 
# p_hat_logit <- predict(glm_fit, newdata = test, type = "response")
# y_hat_logit <- ifelse(p_hat_logit > 0.5, 1, 0) %>% factor
# confusionMatrix(y_hat_logit, test$y)
# # Just as we observed, using just cholestorol, or just one input as a predictor for this
# # somewhat complex disease (or set of disease conditions) is not enough.
# # Since total cholesterol is not that relevant the model just used 
# # As you can see the balanced accuracy is still poor.
# 
# # Try predicting on age only
# glm_fit <- train %>% 
#   glm(y ~ age, data=., family = "binomial")
# 
# p_hat_logit <- predict(glm_fit, newdata = test, type = "response")
# y_hat_logit <- ifelse(p_hat_logit > 0.5, 1, 0) %>% factor
# confusionMatrix(y_hat_logit, test$y)


#####################################################################################
# Model 1: Logistic Regression - with all predictors
#####################################################################################
# Now let's finish up and include all the parameters previously found to be of significance
glm_fit <- train %>% 
  glm(y ~ ., data=., family = "binomial")

p_hat_logit <- predict(glm_fit, newdata = test, type = "response")
y_hat_logit <- ifelse(p_hat_logit > 0.5, 1, 0) %>% factor
cm <- confusionMatrix(y_hat_logit, test$y)
accuracy_results <- tibble(method = "logical regression - all params", accuracy = cm$overall["Accuracy"])
specificity_results <- tibble(method = "logical regression - all params", specificity = cm$byClass["Specificity"])

plot(p_hat_logit)

##################################################################################################
# Model 2: K-Nearest Neighbors (KNN)
##################################################################################################
# Let's try KNN using caret
train_knn <- train(y ~ ., method = "knn", 
                   data = train,
                   tuneGrid = data.frame(k = seq(60, 80, 2)))
train_knn$bestTune
cm <- confusionMatrix(predict(train_knn, test, type = "raw"),
                test$y)
accuracy_results <- bind_rows(accuracy_results, tibble(method = "knn caret", accuracy = cm$overall["Accuracy"]))
specificity_results <- bind_rows(specificity_results, tibble(method = "knn caret", specificity = cm$byClass["Specificity"]))


plot(train_knn)

##################################################################################################
# Model 3: Quadratic Discriminant Analysis (QDA)
##################################################################################################
#train_qda <- train(target ~ ., method = "qda", data = train[ , !(names(train) %in% c("target"))])
train_qda <- train(y ~ ., method = "qda", data = train)
y_hat <- predict(train_qda, test)
cm <- confusionMatrix(data = y_hat, reference = test$y)
accuracy_results <- bind_rows(accuracy_results, tibble(method = "quadratic discriminant analysis (QDA)", accuracy = cm$overall["Accuracy"]))
specificity_results <- bind_rows(specificity_results, tibble(method = "quadratic discriminant analysis (QDA)", specificity = cm$byClass["Specificity"]))

##################################################################################################
# Model 4: Linear Discriminant Analysys (LDA)
##################################################################################################
train_lda <- train(y ~ .,
                   method = "lda",
                   train)
y_hat <- predict(train_lda, test)
cm <- confusionMatrix(data = y_hat, reference = test$y)
accuracy_results <- bind_rows(accuracy_results, tibble(method = "linear discriminant analysis (LDA)", accuracy = cm$overall["Accuracy"]))
specificity_results <- bind_rows(specificity_results, tibble(method = "linear discriminant analysis (LDA)", specificity = cm$byClass["Specificity"]))

##################################################################################################
# Model 5: Decision Tree - using rpart
##################################################################################################
# Decision Tree using rpart
library(rpart)
train_rpart <- train(y ~ ., 
                     method = "rpart",
                     tuneGrid = data.frame(cp = seq(0, 0.5, len = 25)),
                     data = train)

y_hat <- predict(train_rpart, test)
cm <- confusionMatrix(data = y_hat, reference = test$y)
accuracy_results <- bind_rows(accuracy_results, tibble(method = "decision tree (CART)", accuracy = cm$overall["Accuracy"]))
specificity_results <- bind_rows(specificity_results, tibble(method = "decision tree (CART)", specificity = cm$byClass["Specificity"]))

plot(train_rpart)

##################################################################################################
# Model 6: Random Forest
##################################################################################################
library(randomForest)

# plot fit to see what would be a good depth of the tree
fit <- randomForest(y~., data = heart) 
plot(fit)

train_rf <- randomForest(y ~ ., data=train)
cm <- confusionMatrix(predict(train_rf, test), test$y)
accuracy_results <- bind_rows(accuracy_results, tibble(method = "random forest", accuracy = cm$overall["Accuracy"]))
specificity_results <- bind_rows(specificity_results, tibble(method = "random forest", specificity = cm$byClass["Specificity"]))

plot(train_rf)

##################################################################################################
# Final Model Selection
# ---------------------
#
# The best performing are those that have hight accuracy and specificity. These include
#   Logical regression
#   QDA
#   LDA
#   Random forest
#
# QDA has best specificity at the slight relative loss of accuracy
# Random forest has the best accuracy 
# Logistic regression also shows are reasonable combination of accuracy and prediction while
# at the same time being simpler to use with a much better performance compared to some of the
# others.
#
# Of course we can play with hyper-parameters of each model and further pre-process some of the inputs
# to further tweek the numbers up and down.
#
# I pick logistic regression as it is a simpler model and there is no need to use a more complex
# classification model if logistic regression does the trick.
#
# Furthermore since specificity is more important than precision, I tune the model to give more weight
# to specificity rather than precision when training.
##################################################################################################
glm_fit_all <- train %>% 
  glm(y ~ male + age + currentSmoker + cigsPerDay + BPMeds + prevalentStroke + prevalentHyp + diabetes + totChol + sysBP + diaBP + BMI + heartRate, data=., family = "binomial")

p_hat_logit <- predict(glm_fit_all, newdata = test, type = "response")
y_hat_logit <- ifelse(p_hat_logit > 0.1, 1, 0) %>% factor
cm_all <- confusionMatrix(y_hat_logit, test$y)
cm_all
summary(glm_fit_all)

# I see that male, age and sysB are the most contributing inputs towards the training of this model, while
# cigPerDay, diabetes and totlChol also have some contribution. THe other inputs do not shoe any significant
# contribution towards predicting a patient's 10-year chance of getter a heart stroke.

# Let's drop the unused parameters, retrain the model, and check the performance again
glm_fit_reduced <- train %>% 
  glm(y ~ male + age + cigsPerDay + diabetes + totChol + sysBP, data=., family = "binomial")

p_hat_logit <- predict(glm_fit_reduced, newdata = test, type = "response")
y_hat_logit <- ifelse(p_hat_logit > 0.1, 1, 0) %>% factor
cm_reduced <- confusionMatrix(y_hat_logit, test$y)
cm_reduced
summary(glm_fit_reduced)
# THere is a slight but not appreciable decrease. We can train with either option, but it's good to know
# where is the most buck for the benefit, in case we have to train on large amount of data or if we are
# investing a significant amount of money on collecting the inputs that are not useful.


##################################################################################################
## PART B - Comapring Framingham Data Set with UCI Data Set
##################################################################################################
# Load UCI Heart database. Again I downloaded and save the dataset CSV file in my GitHub repository
# fist to make download from R script possible.
myfile <- "https://raw.githubusercontent.com/samseatt/fram-heart/master/data/uci.csv"
uci_heart <- read_csv(myfile)

# Convert the binary output into a factor
uci_heart$target <- as.factor(uci_heart$target)

# Rename output column to y
names(uci_heart)[names(uci_heart) == 'target'] <- 'y'

# Take a peak at the data to understand it and see how it compares to the Framingham study's
# independent variables
head(uci_heart)
str(uci_heart)
summary(uci_heart)

# Partition UCI Heart dataset into traiing and test
set.seed(1)
t_index <- createDataPartition(y = uci_heart$y, times = 1, p = 0.2, list = FALSE)
train <- uci_heart[-t_index,]
test <- uci_heart[t_index,]

# Run all the models

#####################################################################################
# Model 1B: Logistic Regression - with all predictors
#####################################################################################
# Now let's try linear regression again, but train it on all 13 inputs. As I added moore inputs, first things I
# noticed was that specificity also increased. This is important as some

glm_fit <- train %>% 
  glm(y ~ age + ca + chol + cp + oldpeak, data=., family = "binomial")

p_hat_logit <- predict(glm_fit, newdata = test, type = "response")
y_hat_logit <- ifelse(p_hat_logit > 0.5, 1, 0) %>% factor
cm <- confusionMatrix(y_hat_logit, test$y)
uci_accuracy_results <- tibble(method = "logical regression - all params", accuracy = cm$overall["Accuracy"])
uci_specificity_results <- tibble(method = "logical regression - all params", specificity = cm$byClass["Specificity"])

plot(p_hat_logit)

##################################################################################################
# Model 2B: K-Nearest Neighbors (KNN)
##################################################################################################
# Let's try KNN using caret
train_knn <- train(y ~ ., method = "knn", 
                   data = train,
                   tuneGrid = data.frame(k = seq(1, 40, 2)))
train_knn$bestTune
confusionMatrix(predict(train_knn, test, type = "raw"),
                test$y)
uci_accuracy_results <- bind_rows(uci_accuracy_results, tibble(method = "knn caret", accuracy = cm$overall["Accuracy"]))
uci_specificity_results <- bind_rows(uci_specificity_results, tibble(method = "knn caret", specificity = cm$byClass["Specificity"]))

plot(train_knn)

##################################################################################################
# Model 3B: Quadratic Discriminant Analysis (QDA)
##################################################################################################
train_qda <- train(y ~ ., method = "qda", data = train)
y_hat <- predict(train_qda, test)
cm <- confusionMatrix(data = y_hat, reference = test$y)
uci_accuracy_results <- bind_rows(uci_accuracy_results, tibble(method = "quadratic discriminant analysis (QDA)", accuracy = cm$overall["Accuracy"]))
uci_specificity_results <- bind_rows(uci_specificity_results, tibble(method = "quadratic discriminant analysis (QDA)", specificity = cm$byClass["Specificity"]))
# ... The results wit QDA are very good. Similar to logistic regression

##################################################################################################
# Model 4B: Linear Discriminant Analysys (LDA)
##################################################################################################
train_lda <- train(y ~ .,
                   method = "lda",
                   train)
y_hat <- predict(train_lda, test)
cm <- confusionMatrix(data = y_hat, reference = test$y)
uci_accuracy_results <- bind_rows(uci_accuracy_results, tibble(method = "linear discriminant analysis (LDA)", accuracy = cm$overall["Accuracy"]))
uci_specificity_results <- bind_rows(uci_specificity_results, tibble(method = "linear discriminant analysis (LDA)", specificity = cm$byClass["Specificity"]))
# ... Not bad, but the specificity has dropped a little - not perfect for medical diagnostic

##################################################################################################
# Model 5B: Decision Tree - using rpart
##################################################################################################
# Decision Tree using rpart
train_rpart <- train(y ~ ., 
                     method = "rpart",
                     tuneGrid = data.frame(cp = seq(0, 0.5, len = 25)),
                     data = train)

y_hat <- predict(train_rpart, test)
cm <- confusionMatrix(data = y_hat, reference = test$y)
uci_accuracy_results <- bind_rows(uci_accuracy_results, tibble(method = "decision tree (CART)", accuracy = cm$overall["Accuracy"]))
uci_specificity_results <- bind_rows(uci_specificity_results, tibble(method = "decision tree (CART)", specificity = cm$byClass["Specificity"]))

plot(train_rpart)

plot(train_rpart$finalModel, target = 0.1)
text(train_rpart$finalModel, cex = 1)

##################################################################################################
# Model 6B: Random Forest
##################################################################################################
# Random Forest
train_rf <- randomForest(y ~ ., data=train)
cm <- confusionMatrix(predict(train_rf, test), test$y)
uci_accuracy_results <- bind_rows(uci_accuracy_results, tibble(method = "random forest", accuracy = cm$overall["Accuracy"]))
uci_specificity_results <- bind_rows(uci_specificity_results, tibble(method = "random forest", specificity = cm$byClass["Specificity"]))
# ... it's good, but still not as good as Logical Regression

plot(train_rf)

