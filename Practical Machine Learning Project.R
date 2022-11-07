library(caret)
library(tidyr)
library(dplyr)
library(corrplot)
library(naivebayes)
library(e1071)
library(randomForest)

pml <- read.csv("pml-training.csv", na = c("", "NA", "#DIV/0"))

colMeans(is.na(pml))

pml_noNA <- colMeans(is.na(pml))[colMeans(is.na(pml) == 0) == TRUE] %>%
  names()

completepml <- pml[pml_noNA]

inTrain = createDataPartition(completepml$classe, p = .75, list = FALSE)

training = completepml[inTrain,]
testing = completepml[-inTrain,]

dim(training); dim(testing)
head(training)
training <- training[-c(2:5)]
testing <- testing[-c(2:5)]

training_num <- training[,-c(1,2,56)] %>%
  apply(as.numeric, MARGIN = 2)
cor <- cor(training_num)
colinear <- findCorrelation(cor, cutoff = .8, exact = TRUE)
training <- training[,-colinear]
testing <- testing[,-colinear]

training[2:42] %>%
  apply(skewness, MARGIN = 2) %>%
  sort()
par(mfrow = c(2,2))

hist(training$gyros_forearm_y, n = 1000)
hist(training$gyros_forearm_z, n = 1000)
hist(training$gyros_dumbbell_z, n = 1000)

table(round(training$gyros_forearm_y))
max(training$gyros_forearm_y)

training %>%
  filter(training$gyros_forearm_y == 311) #5373
##
table(round(training$gyros_forearm_z))
max(training$gyros_forearm_z)

training %>%
  filter(gyros_forearm_z == 231) #5373
##

table(round(training$gyros_dumbbell_z))
max(training$gyros_dumbbell_z)

training %>%
  filter(training$gyros_dumbbell_z == 317) #5373


training <- training[!(training$X == 5373) == TRUE,]


training[2:42] %>%
  apply(skewness, MARGIN = 2) %>%
  sort()

mod1 <- train(classe ~ ., data = training, preProcess = c("pca", "center", "scale"), method = "naive_bayes")
mod2 <- train(classe ~ ., data = training, preProcess = c("pca", "center", "scale"), method = "treebag")
mod3 <- train(classe ~ ., data = training, preProcess = c("pca","center","scale"), method = "knn")

preObj <- preProcess(training, method = c("pca", "center","scale"))
trainingPC <- predict(preObj, training)
mod4 <- randomForest(as.factor(classe) ~ ., data = trainingPC)

mod5 <- train(classe ~ ., data = training, preProcess = c("center", "scale"), method = "naive_bayes")
mod6 <- train(classe ~ ., data = training, preProcess = c("center", "scale"), method = "treebag")
mod7 <- train(classe ~ ., data = training, preProcess = c("center","scale"), method = "knn")

preObj2 <- preProcess(training, method = c("center","scale"))
trainingPC2 <- predict(preObj2, training)
mod8 <- randomForest(as.factor(classe) ~ ., data = trainingPC2)

predictions1 <- predict(mod1, testing)
predictions2 <- predict(mod2, testing)
predictions3 <- predict(mod3, testing)
testPC <- predict(preObj, testing)
predictions4 <- predict(mod4, testPC)
predictions5 <- predict(mod5, testing)
predictions6 <- predict(mod6, testing)
predictions7 <- predict(mod7, testing)
testPC2 <- predict(preObj2, testing)
predictions8 <- predict(mod8, testPC2)


  
a <- confusionMatrix(as.factor(testing$classe), predictions1)
b <- confusionMatrix(as.factor(testing$classe), predictions2)
c <- confusionMatrix(as.factor(testing$classe), predictions3)
d<- confusionMatrix(as.factor(testing$classe), predictions4)
e <- confusionMatrix(as.factor(testing$classe), predictions5)
f <- confusionMatrix(as.factor(testing$classe), predictions6)
g <- confusionMatrix(as.factor(testing$classe), predictions7)
h <- confusionMatrix(as.factor(testing$classe), predictions8)

a$byClass[1:5,1:2]
b$byClass[1:5,1:2]
c$byClass[1:5,1:2]
d$byClass[1:5,1:2]
e$byClass[1:5,1:2]
f$byClass[1:5,1:2]
g$byClass[1:5,1:2]
h$byClass[1:5,1:2]

testpml <- read.csv("pml-testing.csv", na = c("", "NA", "#DIV/0"))
testpml <- testpml[c(pml_noNA[1:59], "problem_id")]
testpml <- testpml[,-c(2:5)]
testpml <- testpml[,-colinear]


pred1 <- predict(mod1, testpml)
pred2 <- predict(mod2, testpml)
pred3 <- predict(mod3, testpml)
testpmlPC <- predict(preObj, testpml)
pred4 <- predict(mod4, testpmlPC)
pred5 <- predict(mod5, testpml)
pred6 <- predict(mod6, testpml)
pred7 <- predict(mod7, testpml)
testpmlPC2 <- predict(preObj2, testpml)
pred8 <- predict(mod8, testpmlPC2)


comparison <- data.frame(pred1, pred2, pred3, pred4, pred5, pred6, pred7, pred8)
colnames(comparison) <- c("naive_bayes_PCA", "treebag_PCA", "knn_PCA", "rf_PCA", "naive_bayes", "tree_bag", "knn", "rf")
comparison
