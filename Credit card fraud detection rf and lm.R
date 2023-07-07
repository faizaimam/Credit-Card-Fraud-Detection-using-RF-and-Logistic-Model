set.seed(21)
data <- read.csv("creditcard.csv")

summary(data)
str(data)


#Take a look at the dataset
counts <-table(data$Class)
counts
barplot(counts, main = "Fraud Distribution 0: Not Fraud, 1:Fraud", xlab = "Freq")

#The data is heavily imbalanced. So we will scale and sample the data.

#Convert the Class to a factor
data$Class <- as.factor(data$Class)
#Scale
data$Time <- scale(data$Time)
data$Amount <- scale(data$Amount)

#Sampling
library(caTools)
library(ROSE)


sample <- sample.split(data, SplitRatio = 0.8)
train = subset(data, sample == TRUE)
test = subset(data, sample == FALSE)


balancedDataTrain <- ovun.sample(Class~., data=train, method = "under" )
balancedDataTrain<- balancedDataTrain$data
table(balancedDataTrain$Class)
plot(balancedDataTrain$Class)

#Modeling
library(caret)
library(randomForest)

#Random Forest
control <- trainControl(method = "repeatedcv",
                       number =5,
                       repeats = 2)
metric = "Accuracy"
tuneGrid = expand.grid(.mtry = c(1:10))

#Create the model
rf_model <- train(Class~.,
                  data = balancedDataTrain,
                  method = "rf",
                  metric = metric, 
                  tuneGrid = tuneGrid,
                  trControl = control)
varImp(rf_model) 

#Prediction
predict_rf <- predict(rf_model, test)
#ConfusionMatrix
CM_RF <- confusionMatrix(predic_rf, test$Class, positive = "1" )

#ROC and AUC
install.packages("pROC")
library(pROC)
pred_for_ROC <- predict(rf_model, test, type = "prob")
ROC_rf <- roc(test$Class, pred_for_ROC[,2])
ROC_rf_AUC <- auc(ROC_rf)

plot(ROC_rf, main = "ROC for Random Forest (Green)", col = "green")
paste("AUC Random Forest", ROC_rf_AUC)

#Logistic Model
log_model <- glm(Class~., data= balancedDataTrain, family = binomial)
predict_log <- predict(log_model, test, type = "response")

#ROC and AOC
ROC_log <- roc(test$Class, predict_log)
ROC_log_AUC <- auc(ROC_log)

plot(ROC_rf, col = "green", main = "ROC for RF(Green) Vs Logistic Reg(Red)")
lines(ROC_log, col = "red")

paste("AUC Random Forest", ROC_rf_AUC)
paste("AUC Logistic Regression", ROC_log_AUC)




