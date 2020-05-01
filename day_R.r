data = read.csv('day.csv')

summary(data)
str(data)

#Taking care of date
data$dteday = as.Date(data$dteday, "%Y-%m-%d")
data$dteday = as.numeric(format(data$dteday, "%d"))

#Conversion of variables
data$holiday = as.factor(data$holiday)
data$weekday = as.factor(data$weekday)
data$workingday = as.factor(data$workingday)
data$weathersit = as.factor(data$weathersit)
data$season = as.factor(data$season)
data$mnth = as.factor(data$mnth)


#Outlier Analysis usinmg boxplot
boxplot(data$windspeed)
boxplot(data$casual)

num_in = sapply(data, is.numeric)
num_in = data[,num_in]
cname = colnames(num_in)

for(i in cname){
  print(i)
  val = data[,i][data[,i]%in%boxplot.stats(data[,i])$out]
  data = data[which(!data[,i]%in%val),]
}


#Feature selection
corrgram(data[,cname], order = F, upper.panel = panel.pie, text.panel = panel.txt, main = 'Correlation Plot')

data = subset(data, select = -c(instant, atemp))
  
#Splitting into test and train
library('caret')
set.seed(1234)
train.index = createDataPartition(data$cnt,p=0.80,list=FALSE)
train = data[train.index,]
test = data[-train.index,]
  

#Mean Absolute Percentage Error
mape = function(y_true, y_pred){
  mean(abs((y_true-y_pred)/y_true))*100
}


#Decision Tree
library(rpart)
dt_model = rpart(cnt~., data = train, method = 'anova')

dt_predict = predict(dt_model, test[,-14])

mape(test[,14], dt_predict)
#mape = 12.07


#Random Forest
library(randomForest)
rf_model = randomForest(cnt~., train, importance = TRUE, ntree = 300)

rf_predict = predict(rf_model, test[,-14])

mape(test[,14], rf_predict)
#mape = 6.45 



#Linear Regression
lr_test = subset(train, select = c(dteday, yr, mnth, temp, casual, registered, cnt))
lr_model = lm(cnt~., data = lr_test)

lr_predict = predict(lr_model, test[,-14])

mape(test[,14], lr_predict)
#mape = 1.63e-13



#Knn Model
library(class)
knn_model = knnreg(cnt~., train, k=3)

knn_predict = predict(knn_model, test[,-14])

mape(test[,14], knn_predict)
#mape = 1.37
