##################################################################################################
# File: model.R
# Author: Sam Seatt
# Creation Date: May 15, 2019
#
# HarvardX/edX Data Science Capstone - CYO - Modeling and Predicting Heart Disease using the
# Kaggle Framingham Heart study dataset
#
# File Path: code/model.R
# Project Name: fram-heart
# Project GitHub Repository: 
#
##################################################################################################
# This file uses the Framingham Heart study dataset from Kaggle.
# This dataset and the methods and results of the R code in this file are meant to be used as part
# of the CYO (Choose Your Own) project for the HarvardX Data Science Capstone project for
# Sam Seatt (edX id: ).

# The purpose of this R script file is to provides various methods to model, train and capture the
# results of various competing models to analyze this linear classificaton problem
#
# Additionally code is provided to help load the data, clean it, partition it into training and
# validation data sets, statistically analyze it, plot or list it as necessary, save these data sets,
# develop various classificaiton models, train each model on the same trainng set of this data,
# run the predictions to evaluate the results using a handful of parameters, and make snippets
# of these code available for analysis (in the RMD file) for final presentation of the data,
# the methods, and the final results

# This script (model.R) is also available in my RStudio project checked in GitHub at a location listed
# in this file's header.

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
myfile <- "https://raw.githubusercontent.com/samseatt/heart-disease/master/data/heart.csv"
heart <- read_csv(myfile)


##################################################################################################
# Some useful functions and scripts for analyzing and plotting data during data analysis
#
# The code in this section may be used in the rest of this file or inside the final report - please
# refer to the supporting RMD file for the application of these scripts and their extracted results.
#
# Some sections of this code may also depend on data cleansing tasks performed above or 
##################################################################################################


