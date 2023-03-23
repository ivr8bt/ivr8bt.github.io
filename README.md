# Ian Rector Data Science Project Page

# Project 1: RShiny tool utilizing Singscore
This is a tool that I built for my company that utilizes the BioConductor package "Singscore" to rank genes based on their expression in a data set. Genes that are more expressed receive a higher rank. The tool allows the user to input a number for the x number of top genes. For example, if the user were to input 50 then a .xslsx file will be created that shows the 50 highest ranking genes in the dataset. There is an additional component where the user can input a list of genes as a .xlsx file and a .xlsx file will be produced that shows the rank of each of those genes.

# Project 2: Machine Learning on Califonia Lupus Epidemiology Study Dataset
I ran 9 different ML algorithms on the California Lupus Epidemiology Study (CLUES) to distinguish active from inactive disease. These included Naive Baye's, Logistic Regression, Decision Trees, Random Forest, Linear Discriminant Analysis, Adaboost, Gradient Boosting Classifier, K-nearest neighbor classifier and SVM. Unfortunately the results were not great, but almost all lupus datasets have very low sample sizes. However, feature selection using Gini importance yielded interesting results as to the best features for predicting disease activity.  
