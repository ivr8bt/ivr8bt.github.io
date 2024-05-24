# Ian Rector Data Science Project Page

# [Project 1: RShiny tool utilizing Singscore](https://github.com/ivr8bt/Singscore-RSHiny)
This is a tool that I built for my company that utilizes the BioConductor package "Singscore" to rank genes based on their expression in a data set. Genes that are more expressed receive a higher rank. The tool allows the user to input a number for the x number of top genes. For example, if the user were to input 50 then a .xslsx file will be created that shows the 50 highest ranking genes in the dataset. There is an additional component where the user can input a list of genes as a .xlsx file and a .xlsx file will be produced that shows the rank of each of those genes. This is a project that I worked on but the code is not available since it is proprietary information of AMPEL Biosolutions. The code is not publcicly available, but could be demonstrated on request.

# [Project 2: Machine Learning on Califonia Lupus Epidemiology Study Dataset](https://github.com/ivr8bt/CLUES-ML)
I ran 9 different ML algorithms on the California Lupus Epidemiology Study (CLUES) to distinguish active from inactive disease. These included Naive Baye's, Logistic Regression, Decision Trees, Random Forest, Linear Discriminant Analysis, Adaboost, Gradient Boosting Classifier, K-nearest neighbor classifier and SVM. Unfortunately the results were not great, but almost all lupus datasets have very low sample sizes. However, feature selection using Gini importance yielded interesting results as to the best features for predicting disease activity. This is a project that I worked on but the code is not available since it is proprietary information of AMPEL Biosolutions. However, the results and methodology can be obtained [here](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10503349/).

# [Project 3: Regression to Predict German Credit Risk](https://github.com/ivr8bt/German_credit_risk)
Regression on German credit risk. Utilized many different techniques including random forest regression, polynomial regression, linear regression and a fully connected deep neural netowrk. This was a small dataset of only 1,000 indiviudals. Missing values ere imputed using Miss Forest imputation. Unfortunately due to the size of the dataset and other limitations, the best R^2 we achieved was 0.565.

# [Project 4: EDA on European Covid Dataset](https://github.com/ivr8bt/European-Covid)
Data analysis on European Covid data from 2020 through 2022 using R. Conducted exploratory data analysis, hypothesis testing using the Wilcoxon Rank Sum test, explored correlation and conducted autoregression.

# [Project 5: EDA on UNHCR Refugee Data](https://github.com/ivr8bt/UNHCR-Refugee)
EDA on UNHCR data from 1975 through 2021. Examined trends in refugee flows over this time period using data from the UN High Comissioner for Refugees.

# [Project 6: Regression on Ames Housing Dataset](https://github.com/ivr8bt/Ames_Kaggle)
Regression on Ames Housing Dataset on Kaggle. Utilized random forest regression, gradient boosting regression and ridge regression. Achieved results in top the 10% of Kaggle scores.

# [Project 7: Classification of MNIST Images](https://github.com/ivr8bt/MNIST-Classification)
Classification of MNIST data using Deep Neural networks of various sizes and nodes. Train set of 60,000 and test set of 10,000 images. Greatest accuracy on the test set was 0.979.

# [Project 8: Classification of Cifar10 Images](https://github.com/ivr8bt/Cifar10-Classification)
Classification of Cifar10 images using both Deep neural networks and created convolutional neural networks as well as two pretrained networks: ResNet and VGG16. Train set of 50,000 images and test set of 10,000 images. Best test accuracy was 0.735.

# [Project 9: Classification of News Data using AG News Dataset](https://github.com/ivr8bt/AG-News)
Classification of AG News dataset into business, sports, science/tech and world news using simple RNNs, LSTMs and GRUs. Train set of 114,000 articles, validation set of 6,000 articles and test set of 7,600 articles. Able to achieve 86% test accuracy on test set.
