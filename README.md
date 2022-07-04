# Credit Card Fraud Detection
The aim of this project is to predict fraudulent credit card transactions using machine learning models. 

## Project Understanding
Suppose you get a call from your bank, and the customer care executive informs you that your card is about to expire in a week. Immediately, you check your card details and realise that it will expire in the next eight days. Now, to renew your membership, the executive asks you to verify a few details such as your credit card number, the expiry date and the CVV number. Will you share these details with the executive?


In such situations, you need to be careful because the details that you might share with them could grant them unhindered access to your credit card account.
 
In this module, you will understand frauds from a bank’s perspective and learn about the extent to which these frauds affect their business. Banks need to be cautious about their customers’ transactions, as they cannot afford to lose their customers’ money to fraudsters. Every fraud is a loss to the bank, as the bank is responsible for the fraudulent transactions if they are reported within a certain time frame by the customer.

## Data Understanding
As you saw, the data set includes credit card transactions made by European cardholders over a period of two days in September 2013. Out of a total of 2,84,807 transactions, 492 were fraudulent. This data set is highly unbalanced, with the positive class (frauds) accounting for only 0.172% of the total transactions. The data set has also been modified with principal component analysis (PCA) to maintain confidentiality. Apart from ‘time’ and ‘amount’, all the other features (V1, V2, V3, up to V28) are the principal components obtained using PCA. The feature 'time' contains the seconds elapsed between the first transaction in the data set and the subsequent transactions. The feature 'amount' is the transaction amount. The feature 'class' represents class labelling, and it takes the value of 1 in the cases of fraud and 0 in others.

The distribution plots of the variables were Gaussian, which might indicate the effects of transformations that had already occurred on the data set.

So, when you applied PCA on the data set, the new variables were weighted combinations of the original features. 

## Project Pipeline
In the previous segment, you got an intuitive understanding of the data set and understood the type of data that we have and the distribution of each of the variables in the data set. You saw that owing to the PCA transformation, the final variables obtained were normally distributed.

 

The project pipeline can be briefly summarised in the following four steps:

- **Data understanding:** Here, you need to load the data and understand the features present in it. This would help you choose the features that you will need for your final model.
 
- **Exploratory data analytics (EDA):** Usually, in this step, you need to perform univariate and bivariate analyses of the data, followed by feature transformations, if necessary. For the current data set, because Gaussian variables are used, you do not need to perform Z-scaling. However, you can check if there is any skewness in the data and try to mitigate it, as it might cause problems during the model building phase.

Can you think of the reason why skewness can be an issue while modelling? Well, some of the data points in a skewed distribution towards the tail may act as outliers for the machine learning models that are sensitive to outliers; hence, this may cause a problem. Also, if the values of any independent feature are skewed, depending on the model, skewness may affect model assumptions or may impair the interpretation of feature importance. 
 

- **Train/Test split:** Now, you are familiar with the train/test split that you can perform to check the performance of your models with unseen data. Here, for validation, you can use the k-fold cross-validation method. You need to choose an appropriate k value so that the minority class is correctly represented in the test folds.
 
- **Model building / hyperparameter tuning:** This is the final step at which you can try different models and fine-tune their hyperparameters until you get the desired level of performance.


## Class Imbalance
Once you load the data, you can easily observe the disparity between the fraudulent and non-fraudulent cases. In machine learning terms, this situation is known as ‘class imbalance’.

As you observed, the data shows a high class imbalance. Over 2,00,000 cases are mapped to 0, but hardly 500 cases are mapped to 1. Any machine learning algorithm would work well when there is equal representation of each of the classes. However, in this case, regardless of the model that you build, the underlying algorithm will learn more about the non-fraudulent cases rather than the fraudulent ones. Therefore, the loss function optimisation will be heavily biased to the former type of data. This is known as the minority class problem.

 

You can use certain methods to mitigate this problem. They are as follows:

- **Undersampling:** In this method, you have the choice of selecting fewer data points from the majority class for your model building process. If you have only 500 data points in the minority class, you will also have to take 500 data points from the majority class; this will make the classes somewhat balanced. However, in practice, this method is not effective because you will lose over 99% of the original data.
- **Oversampling:** Using this method, you can assign weights to randomly chosen data points from the minority class. In this way, the occurrence of each data point will be multiplied by the assigned weight, and the machine learning algorithm will now be able to focus on this class while optimising the loss function. However, this method does not add any new information and may even exaggerate the existing information quite a bit.

![8766c34f-c95b-41ee-8448-e8545be3bddd-04091e3e-3701-4413-bf6b-ad0e92e15f0c-under vs over](https://user-images.githubusercontent.com/55501944/177137941-c1e7b69f-963c-403b-9c68-91b8900949f6.png)

- **Synthetic Minority Over-Sampling TEchnique (SMOTE):** In this process, you can generate new data points that lie vectorially between two data points that belong to the minority class. These data points are randomly chosen and then assigned to the minority class. This method uses the K-nearest neighbours to create random synthetic samples. The steps in this process are as follows:
1) Randomly selecting a minority point A
2) The k nearest neighbours for that data point belonging to the same are found and then, a random point B form the k_neighbours is selected.
3) Specifying a random value in the range [0, 1] as λ
4) Generating and placing a synthetic sample between the two points A and B on the vector located at λ% from the original point A

![5f413d4e-64d0-4a48-83e0-b78ebdaccebf-SMOTE](https://user-images.githubusercontent.com/55501944/177138209-35818575-155f-4033-a49a-dc388404fddf.png)
                  **New synthetic sample points are added between the two homogenous class points.**

- **ADAptive SYNthetic (ADASYN):** This is similar to SMOTE, with a minor change in the generation of synthetic sample points for minority data points. For a particular data point, the number of synthetic samples that it will add will have a density distribution, whereas for SMOTE, the distribution will be uniform. The aim here is to create synthetic data for minority examples that are harder to learn rather than easier ones. 
 

To sum it up, ADASYN offers the following advantages:

1) It lowers the bias introduced by the class imbalance.
2) It adaptively shifts the classification decision boundary towards difficult examples.

## Introduction to KNN
KNN: K-nearest neighbour is a simple, supervised machine learning algorithm that is used for both classification and regression tasks. It performs these tasks by identifying the neighbours that are the nearest to a data point. For classification tasks, it takes the majority vote and for regression tasks it takes the average value from the neighbours. 

 

The k in KNN specifies the number of neighbours that the algorithm should focus on. For example, if k = 3 then for a particular test data, the algorithm observes the three nearest neighbours and takes the majority vote from them. Depending on the majority of the classes from the three nearby points, the algorithm classifies the test data.



the k value should be an odd number because you have to take the majority vote from the nearest neighbours by breaking the ties. 

Also, note that the k in k-means differs from the k in KNN. k in k-means stands for the number of clusters that can have any number of data points, whereas the k in KNN stands for the number of points that the model would consider to make predictions.

What should the ideal k value be? And what happens if you change it? You will understand this with the help of the following diagrams that show the separation boundary between two classes for different values of 'k'.
![76c1a924-b465-44e5-a3f5-09a071e16699-KNN](https://user-images.githubusercontent.com/55501944/177139227-f168740b-4af5-4be9-9249-64f93aa20343.png)

As you can see, an increase in the k-value causes the decision boundary between the two classes to become smooth. 

- When k = 1 for a data point the model observes the immediate neighbour i.e., it is understanding the noise as well which is causing it to overfit the data. 
- When k = 5 the model observes the five nearest neighbours for a data point. Therefore, the decision boundary starts becoming smooth.
- When k = 10 the model observes the 10 nearest neighbours for a data point. Here, the decision boundary becomes smoother.


## XGBoost
XGBoost stands for ‘eXtreme Gradient Boosting'. It is a decision tree-based ensemble ML algorithm that uses a gradient boosting framework. It is a highly effective and widely used machine learning method and has applications for structured and unstructured data. For structured or tabular data, XGBoost, since its inception, has dominated most of the Kaggle competitions and was even used in challenges such as the Netflix Prize.

![cd97e1fb-fecf-4d33-94ab-a659bf38de1f-Boost](https://user-images.githubusercontent.com/55501944/177139555-336569b3-c44f-4ad5-87c0-09318d46fb22.png)

 

To understand different tree boosting algorithms, let’s recap:

- AdaBoost is an iterative way of adding weak learners to form the final model. For this, each model is trained to correct the errors made by the previous one. This sequential model does this by adding more weight to cases with incorrect predictions. By following this approach, the ensemble model will correct itself while learning by focusing on cases/data points that are hard to predict correctly. 
- Next, we will discuss gradient boosting. In the previous module, you learnt about gradient descent. The same principle applies here as well, wherein the newly added trees are trained to reduce the errors (loss function) made by the earlier models. Overall, in gradient boosting, you are optimising the performance of the boosted model by bringing down the loss function. 
 -XGBoost is an extended version of gradient boosting, wherein it uses more accurate approximations to tune the model and find the best fit. The added features in this are as follows:
  - **Optimisation through second-order derivatives:** The second-order partial derivative of the loss function provides a more detailed picture of the gradient direction. Thus, you can easily find the global minima using the second-order derivative rather than doing it using the first-order derivative.
  - **Advanced regularisation (Lasso and Ridge) to penalise the model based on the number of trees and the depth of the model:** Higher the number of trees, higher will be the number of nodes in each tree and greater will be the penalty attached to it. Whenever you need to add a new node, you will need to check for a minimum reduction in the loss. If there is no significant reduction, you will not create the node.
  - Fast learning through parallel and distributed computing enables quicker model exploration.
 

Owing to parallel processing (speed) and model performance, you can say that XGBoost is gradient boosting on steroids.

## Model Selection and Understanding - I
Now that you have learnt about most of the ML algorithms, you know that there is no single algorithm for solving all problems. You can test out different models on your data and tune them in order to find the best fit. However, sometimes, the steps involved in testing each model for the data will take a lot of computational resources and time. You have to understand the type of data available with you and identify the model that will be the best fit for it. 

 **logistic regression** works best when data is **linearly separable** and **interpretable**. However, the biggest problem occurs mostly when the data has a good amount of **overlap between the classes** present in it. 
 
 As in logistic regression, KNN is also highly interpretable. However, there is a problem while computing the neighbours of a particular test point. You have to find the distance between all the data points and the test data point to come up with the classification of a single neighbour. This task will need a lot of computation and is not a good choice when you have a large amount of data. 
 
 The decision tree model is the first choice when you want the output to be intuitive and want to explain the results to people without a technical background and other stakeholders. You can find the exact node at which a point has been classified and the reason for it. However, when the tree fits all the samples in the training data perfectly, overfitting occurs. Decision trees check the data in many ways. Hence, decision trees tend to overfit if left unchecked.

 

However, working with large data becomes challenging because quadratic computing requires a lot of training time for large data sets.


## Model Selection and Understanding - II

Suppose you are working as a data scientist at a start-up in the healthcare domain. The data you are working on regularly is tabular/structured, and you have comfortably applied the decision tree on it to predict whether the person has skin cancer or not. The available features are not highly correlated such as the skin texture, radius of the skin cells and smoothness. However, now, you decide that you need to work with images rather than the structured data. Using only an image, will the same decision tree be useful for the structured data as well to predict whether the person has skin cancer?

In the situation described above, apart from the insight that pixelated values are highly correlated with each other, you do not have other features available. Since most of the supervised algorithms focus on these features, you will have to use models that can extract features from the present unstructured data.

the thumb rule is that whenever you have structured data, you can use high-performing models such as random forest/XGBoost. However, when you have unstructured data such as an image/text, it is better to use neural networks / LSTM to extract features from the unstructured features.

## Hyperparameter Tuning
You studied model evaluation in Machine Learning-II, in which we focused on hyperparameters and cross-validation. You can consider hyperparameters as model controls/settings. Since the ideal settings of a model used for a particular data set will differ from those of models used for other data sets, you need to tune the model every time you use it to obtain the best results.

Usually, in the machine learning process, you divide the data into train, test and validation sets to evaluate the model’s performance. However, the test and validation sets may increase the variance when the performance of a particular test set might differ from that of another test set.


Also, this hold-out approach (train-test-val) is better when you have enough data points in both classes. However, when the data is imbalanced or less, it is better to use K-Fold Cross Validation for evaluating the performance when the data set is randomly split into ‘k’ groups. Out of these groups, one will be used as the test set, and the remaining groups will be used as train sets. To evaluate the performance, the model will be trained on k-1 groups and then scored using the test set. This process will be iterated until each unique group has been used as the test set.

![08b4ed2b-da1d-445a-8761-75ef519216b4-K fold](https://user-images.githubusercontent.com/55501944/177140634-c1f525ca-eb9a-4908-8c36-50a8a4232500.png)

An extension of K-Fold Cross Validation is Stratified K-Fold Cross Validation, in which you rearrange the data to ensure that each fold is a good representative of all the strata of the data. For imbalanced data, such as the one that we will focus on in this capstone, it is important that the class distribution in each fold is the same as that in the original data set. Stratification ensures that each fold is representative of all the strata of the data.

![1b82c56b-b658-467c-88e9-36a65da856d6-Stratified_k_fold](https://user-images.githubusercontent.com/55501944/177140726-45da56c3-7123-4c66-932f-7b339ba47463.png)
**Stratified K-Fold cross-validation**


Grid Search can be thought of as an exhaustive search of hyperparameters for selecting the ideal hyperparameters for a model. 


We will set up a grid of the hyperparameter values, and for each parameter combination, we will train a model and get a score on the test data. From the hyperparameter value obtained, we will select a nearby range on which the model might perform well. Next, you will take a look at more samples within that range to find the best value within that grid. This is an iterative process, which will continue until we obtain the exact value at which the model is performing the best.

 

Randomised search CV is similar to GridSearchCV but randomly takes samples of parameter combinations from all possible parameter combinations. 

 

When you have a small data set, the computation time will be manageable to test out different hyperparameter combinations. In this scenario, it is advised to use a grid search.

 

However, with large data sets, high dimensions will require a prolonged computation time to train and test each combination. In this scenario, it is advised to use a randomised search because the sampling will be random and not uniform. 

 

For hyperparameter tuning, random and grid search are the two methods available in scikit-learn in the form of RandomiszedSearchCV and GridSearchCV, respectively.

## Model Evaluation

Suppose a person claims that he has built a model for your capstone with 99.83% accuracy. Would you want to productionise that model? He also claims that his model can classify a given transaction as either fraudulent or non-fraudulent with state-of-the-art accuracy. Then you will have a perfect classifier, right? Well, not exactly.

 

This result may sound interesting and even impressive, but we should dive deeper to understand this better. The classes present in the given data set are highly imbalanced, with 99.83% of the observations being labelled as non-fraudulent transactions and only 0.17% of the observations being labelled as fraudulent. So, without handling the imbalances present, the model overfits on the training data and therefore classifying every transaction as non-fraudulent; hence, it achieves the aforementioned accuracy.


 accuracy is not always the correct metric for solving classification problems. There are other metrics such as precision, recall, confusion matrix, F1 score and the AUC-ROC score.
 



The ROC curve is used to understand the strength of the model by evaluating the performance of the model at all the classification thresholds.



 the default threshold of 0.5 is not always the ideal threshold to find the best classification label of the test point. Since the ROC curve is measured at all thresholds, the best threshold would be one at which the TPR is high and FPR is low, i.e., misclassifications are low.


After determining the optimal threshold, you can calculate the F1 score of this classifier to measure the precision and recall at the selected threshold. 



Finding the best F1 score is not the last step. This score depends on both precision and recall. So, depending on the use case, you have to account for what you need–high precision or high recall.

 

You learnt how the model was a bad classifier when it labelled every transaction as non-fraudulent with a high accuracy. But what if the scenario is the opposite?


If the model labels all data points as fraudulent, then your recall becomes 1.0. However, at the same time, your precision is compromised. Precision is the ability of a classification model to identify only the relevant data points. When you increase the recall, you will also decrease the precision.

 

You can maximise the recall or precision at the expense of another metric, which depends on whether you want a high precision or a high recall for your use case.


For banks with a smaller average transaction value, you would want a high precision because you only want to label relevant transactions as fraudulent. For every transaction that is flagged as fraudulent, you can add the human element to verify whether the transaction was made by calling the customer. However, when the precision is low such tasks are a burden because the human element has to be increased.


For banks having a larger transaction value, if the recall is low, i.e., it is unable to detect transactions that are labelled as non-fraudulent. So, consider the losses if the missed transaction was a high-value fraudulent one, for example, a transaction of $10,000?


Here, to save banks from high-value fraudulent transactions, we need to focus on a high recall to detect actual fraudulent transactions.


Now, you are ready to build your model for this capstone. In the next session, we will explain the problem statement, deliverables, deadlines, etc. of the project.



### Summary

You have come a long way! You have learnt different techniques in machine learning using which you should now be able to build your own model for credit card fraud detection.

The most important points to re-iterate are as follows:

 

- Class imbalances:

  - In undersampling, you select fewer data points from the majority class for your model building process to balance both classes.
  - In oversampling, you assign weights to randomly chosen data points from the minority class. This is done so that the algorithm can focus on this class while optimising the loss function.
  - SMOTE is a process using which you can generate new data points that lie vectorially between two data points that belong to the minority class.
  - ADASYN is similar to SMOTE, with a minor change in the sense that the number of synthetic samples that it will add will have a density distribution. The aim here is to create synthetic data for minority examples that are harder to learn rather than the easier ones. 
 

- Model selection and understanding:

  - Logistic regression works best when the data is linearly separable and needs to be interpretable. 
  - KNN is also highly interpretable but not preferred when you have a huge amount of data, as it will consume a lot of computation.
  - The decision tree model is the first choice when you want the output to be intuitive, but they tend to overfit if left unchecked.
  - KNN is a simple, supervised machine learning algorithm that is used for both classification and regression tasks. The k value in KNN should be an odd number because you have to take the majority vote from the nearest neighbours by breaking the ties. 
  - In Gradient Boosted machines/trees, newly added trees are trained to reduce the errors (loss function) of earlier models.
  - XGBoost is an extended version of gradient boosting, with additional features such as regularisation and parallel tree learning algorithm for finding the best split. 


- Hyperparameter tuning:

  - When the data is imbalanced or less, it is better to use K-Fold Cross Validation for evaluating the performance when the data set is randomly split into ‘k’ groups.
  - Stratified K-Fold Cross Validation is an extension of K-Fold cross-validation, in which you rearrange the data to ensure that each fold is a good representative of all the strata of the data.
  - When you have a small data set, the computation time will be manageable to test out different hyperparameter combinations. In this scenario, it is advised to use a grid search.
  - However, with large data sets, it is advised to use a randomised search because the sampling will be random and not uniform. 
 

- Model evaluation:

  - Accuracy is not always the correct metric for solving classification problems of imbalanced data.
  - Since the ROC curve is measured at all thresholds, the best threshold would be one at which the TPR is high and FPR is low, i.e., misclassifications are low.
  - Depending on the use case, you have to account for what you need–high precision or high recall.




# Problem Statement
The problem statement chosen for this project is to predict fraudulent credit card transactions with the help of machine learning models.

 

In this project, you will analyse customer-level data that has been collected and analysed during a research collaboration of Worldline and the Machine Learning Group. 

 

The data set is taken from the Kaggle website and has a total of 2,84,807 transactions; out of these, 492 are fraudulent. Since the data set is highly imbalanced, it needs to be handled before model building.

 

### Business problem overview
For many banks, retaining high profitable customers is the number one business goal. Banking fraud, however, poses a significant threat to this goal for different banks. In terms of substantial financial losses, trust and credibility, this is a concerning issue to both banks and customers alike.


It has been estimated by Nilson Report that by 2020, banking frauds would account for $30 billion worldwide. With the rise in digital payment channels, the number of fraudulent transactions is also increasing in new and different ways. 

 

In the banking industry, credit card fraud detection using machine learning is not only a trend but a necessity for them to put proactive monitoring and fraud prevention mechanisms in place. Machine learning is helping these institutions to reduce time-consuming manual reviews, costly chargebacks and fees as well as denials of legitimate transactions.

 

### Understanding and defining fraud
Credit card fraud is any dishonest act or behaviour to obtain information without proper authorisation from the account holder for financial gain. Among different ways of committing frauds, skimming is the most common one, which is a way of duplicating information that is located on the magnetic strip of the card. Apart from this, following are the other ways:

  - Manipulation/alteration of genuine cards
  - Creation of counterfeit cards
  - Stealing/loss of credit cards
  - Fraudulent telemarketing
 

### Data dictionary
The data set can be downloaded using this link.

 

The data set includes credit card transactions made by European cardholders over a period of two days in September 2013. Out of a total of 2,84,807 transactions, 492 were fraudulent. This data set is highly unbalanced, with the positive class (frauds) accounting for 0.172% of the total transactions. The data set has also been modified with principal component analysis (PCA) to maintain confidentiality. Apart from ‘time’ and ‘amount’, all the other features (V1, V2, V3, up to V28) are the principal components obtained using PCA. The feature 'time' contains the seconds elapsed between the first transaction in the data set and the subsequent transactions. The feature 'amount' is the transaction amount. The feature 'class' represents class labelling, and it takes the value of 1 in cases of fraud and 0 in others.

 

### Project pipeline
The project pipeline can be briefly summarised in the following four steps:

  - Data Understanding: Here, you need to load the data and understand the features present in it. This would help you choose the features that you will need for your final model.
  - Exploratory data analytics (EDA): Normally, in this step, you need to perform univariate and bivariate analyses of the data, followed by feature transformations, if necessary. For the current data set, because Gaussian variables are used, you do not need to perform Z-scaling. However, you can check whether there is any skewness in the data and try to mitigate it, as it might cause problems during the model building phase.
  - Train/Test split: Now, you are familiar with the train/test split that you can perform to check the performance of your models with unseen data. Here, for validation, you can use the k-fold cross-validation method. You need to choose an appropriate k value so that the minority class is correctly represented in the test folds.
  - Model building / hyperparameter tuning: This is the final step at which you can try different models and fine-tune their hyperparameters until you get the desired level of performance on the given data set. You should try and check if you get a better model by various sampling techniques.
  - Model evaluation: Evaluate the models using appropriate evaluation metrics. Note that since the data is imbalanced, it is more important to identify the fraudulent transactions accurately than the non-fraudulent ones. Choose an appropriate evaluation metric that reflects this business goal.
