# Ian Rector Data Science Portfolio Page

# About Me
My name is Ian Rector and I am a recent graduate of the masters of science in data science program at Northwestern University. Prior to that I double majored in computer science and biology at the University of Virginia and went on to work as a Bioinformatics analyst at AMPEL BioSolutions in Charlottesville. At AMPEL I analyzed large genetic datasets (RNA-Seq and microarray primarily) of Lupus patients using R. I also used various supervised learning techniques to predict the disease activity of these patients and used unsupervised learning to create Lupus patient clusters, which is necessary in a heterogenous disease like Lupus in Python. I greatly enjoyed my time there and it led to a passion for working with data. I enrolled at Northwestern to further bolster my data science skills. Here are seven different projects that are only a small sample of my data science skills. Here I show a classification model that I created for a first author paper that I worked on at AMPEL, a comparative analysis of pre-built convolutional neural network models, classification of news data using reccurrent neural networks, LSTMs and GRUS, classification of the MNIST dataset using deep neural networks, regression using random forest regression, gradient boosting regression and ridge regression as well as EDA on UNHCR refugee data using Python and EDA on European Covid data using R.

# [Project 1: Machine Learning on Califonia Lupus Epidemiology Study Dataset](https://github.com/ivr8bt/CLUES-ML)
I ran 9 different supervised learning algorithms on the California Lupus Epidemiology Study (CLUES) to distinguish active from inactive disease. These included Naive Baye's, Logistic Regression, Decision Trees, Random Forest, Linear Discriminant Analysis, Adaboost, Gradient Boosting Classifier, K-nearest neighbor classifier and SVM. The Decsion Tree was found to be the best model and achieved an accuracy of 85% as well as a specificty of 95% and precision of 92% on the test set. Feature selection using Gini importance showed the top features for predicting disease activity, which was very notable in this case since figuring out the top features for predicitng disease activity is more imrotant than the model since model's predicting SLE will often be limited. This is a paper that I authored but the code is not available since it is proprietary information of AMPEL Biosolutions. However, the results and methodology can be obtained [here](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10503349/).

![](/images/Figure%206.png)

# [Project 2: CNN Pre-built Architecture Analysis](https://github.com/ivr8bt/CNN-Architecture-Analysis)
I analyze the LeNet5, VGG16, InceptionV3 and ResNet50 architectures on 4 different datasets: Cifar10, Cats vs. Dogs Redux: Kernel addition, Eurosat and Horses vs. Humans. Each of these datasets involve different kinds of comparison. 2 are multiclass and 2 are binary classification problems. The datasets also have different image sizes and numbers of images. I found that VGG16 and InnceptionV3 performed the best across all 4 datasets. InceptionV3 was overall the best since it had comparable accuracy to VGG16 with an average accuracy of 86% across the four datasets; however, it was significantly faster than VGG16. Below are plots depicting the training and validation loss and accuracy for the VGG16 model on the dogs versus cats dataset:

![](/images/Training%20Accuracy%20for%20Dogs%20vs%20Cats.png)

# [Project 3: Classification of News Data using AG News Dataset](https://github.com/ivr8bt/AG-News)
Classification of AG News dataset into business, sports, science/tech and world news using simple RNNs, LSTMs and GRUs. Train set of 114,000 articles, validation set of 6,000 articles and test set of 7,600 articles. Able to achieve 86% test accuracy on test set.

![](/images/Results%20Table.png)

# [Project 4: Predicting German Credit Risk](https://github.com/ivr8bt/German_credit_risk)
The primary objective of this project is to develop a predictive model using the German credit risk dataset that can accurately forecast the credit amount that potential borrowers are likely to receive.
The dataset only consisted of 1000 people, so after EDA we decided to use the missforest imputation method to fill in missing data. We evaluated 3 different models: simpl linear regression, random forest regression and a deep neural network and found that the DNN performed the best with a R^2 value of 0.537.


# [Project 5: Classification of MNIST Images](https://github.com/ivr8bt/MNIST-Classification)
Classification of MNIST data using Deep Neural networks of various sizes and nodes. Train set of 60,000 and test set of 10,000 images. Greatest accuracy on the test set was 0.979. Below is a plot showing the explained variance by number of PCA components for the 784 pixel features in the MNIST dataset.

![](/images/Explained%20Variance%20ratio%20for%20PCA.png)

# [Project 6: EDA on European Covid Dataset](https://github.com/ivr8bt/European-Covid-2020-2022)
Data analysis on European Covid data from 2020 through 2022 using R. I conducted exploratory data analysis, hypothesis testing using the Wilcoxon Rank Sum test, explored correlation and conducted autoregression. I looked at the rate of covid incidence and infection among all European countries and then focused in on France and Germany to identify trends in infection and incidence rates over time in the two countries.

![](/images/Covid%20Deaths%20over%20time.png)

# [Project 7: EDA on UNHCR Refugee Data](https://github.com/ivr8bt/UNHCR-Refugee)
Exploratory Data Analysis examining refugee flows 1975-2021 using data from the UN High Comissioner for Refugees. I identified the top countries where refugees flee from and where they go to by decade as well as cumulatively.

![](/images/Refugees%20by%20country.png)

# [Project 8: Regression on Ames Housing Dataset](https://github.com/ivr8bt/Ames_Kaggle)
Regression on Ames Housing Dataset on Kaggle. Utilized random forest regression, gradient boosting regression and ridge regression. Achieved results in the top 10% of Kaggle scores.

![](/images/Correlation%20Heatmap%20for%20Ames%20Housing.png)
