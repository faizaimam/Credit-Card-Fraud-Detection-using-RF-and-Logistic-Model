This project was done to get a better understanding of the topic by doing the work practically. So, there might be mistakes or misinterpretations. Feel free to point them out. Thank you!
## Credit Card Fraud Detection (Random Forest Model and Logistic Model)


### 1. Packages

```{r eval =TRUE, warning = FALSE, message = FALSE}
library(caTools)
library(ROSE)
library(caret)
library(randomForest)
library(pROC)
```

### 2. Data Description

The data used here is from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud.

This dataset contains transactions made by credit cards in September 2013 by European cardholders. The dataset has been collected and analysed during a research collaboration of Worldline and the Machine Learning Group (http://mlg.ulb.ac.be) of ULB (Université Libre de Bruxelles) on big data mining and fraud detection.

This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

It contains only numerical input variables which are the result of a PCA transformation. Due to confidentiality issues, the original features and more background information about the data could not be obtained. Features `V1`,`V2`, … `V28` are the principal components obtained with PCA, the only features which have not been transformed with PCA are `Time`and `Amount`. Feature `Time` contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature `Amount` is the transaction Amount, this feature can be used for example-dependent cost-sensitive learning. Feature `Class` is the response variable and it takes value 1 in case of fraud and 0 otherwise.


#### 2.1 Loading Data

```{r comment = ""}
data <-read.csv("creditcard.csv", header =T)
```

#### 2.2 Looking at the Dataset

```{r comment=""}
counts <-table(data$Class)
counts
barplot(counts, main = "Fraud Distribution 0: Not Fraud, 1:Fraud", xlab = "Freq")
```
```

     0      1 
284315    492
```

![Barplot](https://github.com/faizaimam/Credit-Card-Fraud-Detection-using-RF-and-Logistic-Model/assets/73066502/6d9aff20-1188-48ab-8fe5-4f292103b683)

As we can see from both the table and barplot, the dataset is highly unbalanced. Number of Normal cases is way larger than number of Fraud cases. This is why we have to balance the dataset. 


#### 2.3 Preprocessing Data
Taking the `Class`variable transforming it from integer to factor with levels- "1" and "0" as Fraud and Normal case respectively.

```{r}
data$Class <- as.factor(data$Class)
head(data)
```

#### 2.4 Scaling Necessary Variables
As Time and Amount has different measuring units, we have to scale the variables for correct analysis.


```{r}
data$Time <- scale(data$Time)
data$Amount <- scale(data$Amount)
```

#### 2.5 Using sampling to balance the data
We use undersampling to balance the imbalance in data. Undersampling is a technique to balance uneven datasets by keeping almost all of the data in the minority class and decreasing the size of the majority class
```{r comment = ""}
set.seed(73)
sample <- sample.split(data, SplitRatio = 0.8)
train = subset(data, sample == TRUE)
test = subset(data, sample == FALSE)

balancedDataTrain <- ovun.sample(Class~., data=train, method = "under" )
balancedDataTrain<- balancedDataTrain$data
table(balancedDataTrain$Class)
plot(balancedDataTrain$Class)
```
```

  0   1 
377 387
```
![Barplot of balanced data](https://github.com/faizaimam/Credit-Card-Fraud-Detection-using-RF-and-Logistic-Model/assets/73066502/ef420403-b481-4329-bfe1-fadd8e9fcf2a)

Now that number of Normal cases and Fraud cases have been balanced, we proceed to do further analysis.

### 3. Modeling

#### 3.1 Random Forest Model
Random Forest is a popular machine learning algorithm and is particularly useful when dealing with complex and high-dimensional datasets. The model is an ensemble learning method that combines multiple decision trees, known as the "forest," to make more accurate and robust predictions. Each decision tree in the forest is trained on a random subset of the data and a random subset of features, which helps to reduce overfitting and improve generalization.
```{r comment = "" }
control <- trainControl(method = "repeatedcv",
                        number =5,
                        repeats = 2)
metric = "Accuracy"
tuneGrid = expand.grid(.mtry = c(1:10))

#Creating the model
rf_model <- train(Class~.,
                  data = balancedDataTrain,
                  method = "rf",
                  metric = metric, 
                  tuneGrid = tuneGrid,
                  trControl = control)
```
The variables that are more important in this Random Forest Model-
```{r comment = ""}
varImp(rf_model)
```
```
rf variable importance

  only 20 most important variables shown (out of 30)

        Overall
V14    100.0000
V10     69.6611
V4      37.3923
V17     35.9881
V12     32.9291
V11     27.5330
V3      12.5153
V7       7.2420
V16      7.0554
Amount   4.1081
V9       3.9608
V2       3.8815
V8       3.3982
V21      3.2488
V19      2.9260
V20      2.3368
V13      1.1545
V6       1.0730
V15      1.0339
V23      0.9979
```
##### 3.1.1 Confusion Matrix of RF Model
A confusion matrix is primarily employed to evaluate the performance of a classification model. It is a square matrix that compares the predicted classifications of the model with the actual labels of the data. The matrix organizes the outcomes into four categories: true positives (correctly predicted positive instances), true negatives (correctly predicted negative instances), false positives (incorrectly predicted positive instances), and false negatives (incorrectly predicted negative instances). By examining these elements, the confusion matrix provides valuable insights into the model's accuracy, precision, recall, and F1 score, which are crucial performance metrics. This allows us to gain a comprehensive understanding of how well our model is performing and make informed decisions on potential improvements and adjustments.

```{r comment="" }
predict_rf <- predict(rf_model, test)
CM_RF <- confusionMatrix(predict_rf, test$Class, positive = "1" )
CM_RF

```
```
Confusion Matrix and Statistics

          Reference
Prediction     0     1
         0 62835    11
         1  1370    94
                                          
               Accuracy : 0.9785          
                 95% CI : (0.9774, 0.9796)
    No Information Rate : 0.9984          
    P-Value [Acc > NIR] : 1               
                                          
                  Kappa : 0.1171          
                                          
 Mcnemar's Test P-Value : <2e-16          
                                          
            Sensitivity : 0.895238        
            Specificity : 0.978662        
         Pos Pred Value : 0.064208        
         Neg Pred Value : 0.999825        
             Prevalence : 0.001633        
         Detection Rate : 0.001462        
   Detection Prevalence : 0.022765        
      Balanced Accuracy : 0.936950        
                                          
       'Positive' Class : 1
```
##### 3.1.2 Receiver Operating Characteristic (ROC) and Area Under the Curve (AUC)
```{r message = FALSE, comment =""}
pred_for_ROC <- predict(rf_model, test, type = "prob")
ROC_rf <- roc(test$Class, pred_for_ROC[,2])
ROC_rf_AUC <- auc(ROC_rf)

plot(ROC_rf, main = "ROC for Random Forest (Green)", col = "green")
paste("AUC Random Forest", ROC_rf_AUC)
```
![roc rf plot](https://github.com/faizaimam/Credit-Card-Fraud-Detection-using-RF-and-Logistic-Model/assets/73066502/806dde07-d191-4bf7-9ccb-f0621a511698)
```
[1] "AUC Random Forest 0.973596923544747"
```

#### 3.2 Logistic Model
The logistic model is commonly used in various fields, most notably in statistics and machine learning, to predict binary outcomes. It is particularly effective when dealing with situations where the dependent variable is categorical and has two possible outcomes, often labeled as "success" or "failure," "yes" or "no," or "1" or "0." The logistic model employs the logistic function, which maps any real-valued input to an output between 0 and 1, representing the probability of the binary event occurring.
```{r warning = FALSE, message =FALSE, comment= ""}
log_model <- glm(Class~., data= balancedDataTrain, family = binomial)
predict_log <- predict(log_model, test, type = "response")
```

##### 3.2.1 Receiver Operating Characteristic (ROC) and Area Under the Curve (AUC)
```{r warning = FALSE,message = FALSE, comment = "" }
ROC_log <- roc(test$Class, predict_log)
ROC_log_AUC <- auc(ROC_log)
plot(ROC_log, main = "ROC for Logistic Reg(Red)", col = "red")
paste("AUC Logistic Regression", ROC_log_AUC)
```
![ROC Log Plot](https://github.com/faizaimam/Credit-Card-Fraud-Detection-using-RF-and-Logistic-Model/assets/73066502/592fe0c6-33a8-4acd-a2a1-252145fe50bd)
```
[1] "AUC Logistic Regression 0.966075776623242"
```
###  4. Comparison between Random Forest and Logistic Model
```{r comment = ""}
plot(ROC_rf, col = "green", main = "ROC for RF(Green) Vs Logistic Reg(Red)")
lines(ROC_log, col = "red")
paste("AUC Random Forest is", ROC_rf_AUC, "and AUC Logistic Regression is", ROC_log_AUC)

```
![comparison plot](https://github.com/faizaimam/Credit-Card-Fraud-Detection-using-RF-and-Logistic-Model/assets/73066502/da9e3f70-5f96-4062-9fd3-c928a3d88661)
```
 "AUC Random Forest is 0.973596923544747 and AUC Logistic Regression is 0.966075776623242"
```
As AUC value is slightly higher in Random Forest Model than Logistic Model, Random Forest Model is supposed to be the better model for Credit Card Fraud detection based on this dataset. However, the difference between the value is very small, so either model will suffice.
