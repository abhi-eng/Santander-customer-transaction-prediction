##################     PROJECT: 2) Satander Customer Transaction Prediction  Project     ##################   ##################   ##################

#Remove the elements
rm(list = ls())

#Set working directory
setwd("/home/nupur/Downloads/edwisor")

#Check working directory
getwd()

library(ggplot2)
library(FactoMineR)
library(tidyverse)
library(moments)
library(DataExplorer)
library(caret)
library(Matrix)
library(pdp)
library(mlbench)
library(caTools)
library(randomForest)
library(glmnet)
library(mlr)
library(vita)
library(rBayesianOptimization)
library(lightgbm)
library(pROC)
library(DMwR)
library(ROSE)
library(yardstick)
library(DataCombine)
library(rpart)
library(usdm)

# loading datasets
train = read.csv("train.csv",header=T,na.strings = c(""," ","NA"))

test = read.csv("test.csv",header=T,na.strings = c(""," ","NA"))

##############################  Exploratory data analysis ##########################################

#Getting the column names of the dataset
colnames(train)
colnames(test)

#Getting the number of variables and obervation in the datasets
dim(train)
dim(test)

# Structure of data
str(train)
str(test)

#Summary of datasets
summary(train)
summary(test)

#look at top 5 observations
head(train,5)
head(test,5)

#data type of variables
sapply(train,class)

#changing datatype of target variable to factor datatype.
train$target<-as.factor(train$target)

#Percenatge counts of target classes
table(train$target)/length(train$target)*100

#Bar plot for count of target classes
plot1<-ggplot(train,aes(target))+theme_bw()+geom_bar(stat='count',fill='lightgreen')

##We have a unbalanced data,where 90% of the data is the data of number of customers those will not make a transaction and 10% of the data is those who will make a transaction.

#checking for duplicates
dup<-function(x){if(length(unique(colnames(x))==ncol(x))){print('No')}else{print('Yes')}}
cat('Is there any duplicate column in train data:', dup(train), 
    '\nIs there any duplicate column in test data:', dup(test), sep = ' ') 

### No duplicates

#Visulisation
#Distribution of train attributes from 3 to 102
for (var in names(train_df)[c(3:102)]){
  target<-train_df$target
  plot<-ggplot(train_df, aes(x=train_df[[var]],fill=target)) +
    geom_density(kernel='gaussian') + ggtitle(var)+theme_classic()
  print(plot)
}

#Distribution of train attributes from 103 to 202
for (var in names(train_df)[c(103:202)]){
  target<-train_df$target
  plot<-ggplot(train_df, aes(x=train_df[[var]], fill=target)) +
    geom_density(kernel='gaussian') + ggtitle(var)+theme_classic()
  print(plot)
}

#Distribution of test attributes from 2 to 101
plot_density(test_df[,c(2:101)], ggtheme = theme_classic(),geom_density_args = list(color='blue'))

#Distribution of test attributes from 102 to 201
plot_density(test_df[,c(102:201)], ggtheme = theme_classic(),geom_density_args = list(color='blue'))


##################################  DATA PREPROCESSING     ###############################################
          ######################## Missing Values Analysis #####################################

#checking for missing values
missing_val = data.frame(apply(train,2,function(x){sum(is.na(x))}))

#creating new column contains name of columns
missing_val$Columns = row.names(missing_val)

#changing name of column containg missing values
names(missing_val)[1] =  "Missing_percentage"

#changing missing values to missing percentages
missing_val$Missing_percentage = (missing_val$Missing_percentage/nrow(train)) * 100

#arrange missing percentage in descending order
missing_val = missing_val[order(-missing_val$Missing_percentage),]

#removing row names to index
row.names(missing_val) = NULL

#interchange columns
missing_val = missing_val[,c(2,1)]

## No missing values present.

############################## OUTLIER ANALYSIS ##########################################

#Outlier analysis for test dataset.
numeric_index = sapply(train,is.numeric) #selecting only numeric

numeric_data = train[,numeric_index]

cnames = colnames(numeric_data)

for (i in 1:length(cnames))
   {
     assign(paste0("gn",i), ggplot(aes_string(y = (cnames[i]), x = "target"), data = subset(train))+ 
              stat_boxplot(geom = "errorbar", width = 0.5) +
              geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
                           outlier.size=1, notch=FALSE) +
              theme(legend.position="bottom")+
              labs(y=cnames[i],x="target")+
              ggtitle(paste("Box plot of target for",cnames[i])))
}

#visualise some variables
gridExtra::grid.arrange(gn3,gn4,gn5,ncol=3)
gridExtra::grid.arrange(gn6,gn7,gn8,ncol=3)
gridExtra::grid.arrange(gn9,gn10,gn11,ncol=3)

# #loop to remove outliers from  train dataset variables
 for(i in cnames){
   print(i)
   val = train[,i][train[,i] %in% boxplot.stats(train[,i])$out]
   #print(length(val))
   train = train[which(!train[,i] %in% val),]
 }
numeric_index = sapply(test,is.numeric) #selecting only numeric

numeric_data = test[,numeric_index]

cnames = colnames(numeric_data)

# #loop to remove outliers from test dataset variables
for(i in cnames){
  print(i)
  val = test[,i][test[,i] %in% boxplot.stats(test[,i])$out]
  #print(length(val))
  test = test[which(!test[,i] %in% val),]
}

############################## FEATURE SELECTION ##########################################

#Correlations in train data
#convert factor to int.
train$target<-as.numeric(train$target)
train_correlations<-cor(train[,c(2:202)])
train_correlations
##We can observed that the correlation between the train attributes is very small.

#Correlations in test data
test_correlations<-cor(test[,c(2:201)])
test_correlations
##We can observed that the correlation between the test attributes is very small.

###################################Model Development#######################################

#Split the data using CreateDataPartition
set.seed(689)
#train.index<-createDataPartition(train_df$target,p=0.8,list=FALSE)
train.index<-sample(1:nrow(train),0.8*nrow(train))
#train data
train.data<-train[train.index,]
#validation data
valid.data<-train[-train.index,]
#dimension of train data
dim(train.data)
#dimension of validation data
dim(valid.data)
#target classes in train data
table(train.data$target)
#target classes in validation data
table(valid.data$target)
table(train$target)

#####################  Decision tree for classification  #####################################
#Develop Model on training data
C50_model = C5.0(target ~., train, trials = 100, rules = TRUE)

#Summary of DT model
summary(C50_model)

#write rules into disk
write(capture.output(summary(C50_model)), "c50Rules.txt")

#Lets predict for test cases
C50_Predictions = predict(C50_model, test[,-2], type = "class")

##Evaluate the performance of classification model
ConfMatrix_C50 = table(test$target, C50_Predictions)
confusionMatrix(ConfMatrix_C50)
error_metric(ConfMatrix_C50)
#Accuracy: 0.87

# Area under ROC curve
roc.curve(test$target,C50_Predictions)
#Area under the curve : 0.53

########################## Random Forest ####################################################
RF_model = randomForest(target ~ ., train, importance = TRUE, ntree = 500)

#Predict test data using random forest model
RF_Predictions = predict(RF_model, test[,-17])

##Evaluate the performance of classification model
ConfMatrix_RF = table(test$target, RF_Predictions)
confusionMatrix(ConfMatrix_RF)
error_metric(ConfMatrix_RF)
# accuracy = 0.88
roc.curve(test$target,RF_Predictions)
#Area under the curve for train_:0.54

######################################## Logistic Regression ###############################
logit_model = glm(target ~ ., data = train, family = "binomial")

#summary of the model
summary(logit_model)

#predict using logistic regression
logit_Predictions = predict(logit_model, newdata = test, type = "response")

#convert prob
logit_Predictions = ifelse(logit_Predictions > 0.5, 1, 0)


##Evaluate the performance of classification model
ConfMatrix_RF = table(test$target, logit_Predictions)

ConfMatrix_LR
error_metric(ConfMatrix_LR)
# accuracy = 0.9
roc.curve(test$target,logit_Predictions>0.5)
#Area under the curve for train: 0.78

############################## lightgbm #################################################
#Convert data frame to matrix
set.seed(5432)
X_train<-as.matrix(train.data[,-c(1,2)])
y_train<-as.matrix(train.data$target)
X_valid<-as.matrix(valid.data[,-c(1,2)])
y_valid<-as.matrix(valid.data$target)
test_data<-as.matrix(test[,-c(1)])

#training data
lgb.train <- lgb.Dataset(data=X_train, label=y_train)
#Validation data
lgb.valid <- lgb.Dataset(data=X_valid,label=y_valid)

#Selecting best hyperparameters
set.seed(653)
lgb.grid = list(objective = "binary",
                metric = "auc",
                boost='gbdt',
                max_depth=-1,
                boost_from_average='false',
                min_sum_hessian_in_leaf = 12,
                feature_fraction = 0.05,
                bagging_fraction = 0.45,
                bagging_freq = 5,
                learning_rate=0.02,
                tree_learner='serial',
                num_leaves=20,
                num_threads=5,
                min_data_in_bin=150,
                min_gain_to_split = 30,
                min_data_in_leaf = 90,
                verbosity=-1,
                is_unbalance = TRUE)

#Training the lgbm model
set.seed(7663)
lgbm.model <- lgb.train(params = lgb.grid, data = lgb.train, nrounds =10000,eval_freq =1000,
                        valids=list(val1=lgb.train,val2=lgb.valid),early_stopping_rounds = 5000)

#lgbm model performance on test data
set.seed(6532)
lgbm_pred_prob <- predict(lgbm.model,test_data)
print(lgbm_pred_prob)
#Convert to binary output (1 and 0) with threshold 0.5
lgbm_pred<-ifelse(lgbm_pred_prob>0.5,1,0)
print(lgbm_pred)

set.seed(6521)
#feature importance plot
tree_imp <- lgb.importance(lgbm.model, percentage = TRUE)
lgb.plot.importance(tree_imp, top_n = 50, measure = "Frequency", left_margin = 10)

##lightgbm is performing well on imbalanced data compared to other models based on scores of roc_auc_score.

##Final submission
sub_df<-data.frame(ID_code=test_df$ID_code,lgb_predict_prob=lgbm_pred_prob,lgb_predict=lgbm_pred)
write.csv(sub_df,'submission.CSV',row.names=F)
head(sub_df)

