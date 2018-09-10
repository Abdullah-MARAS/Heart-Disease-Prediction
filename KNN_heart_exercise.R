
#Adapted from Elif Kartal
#Diagnosis of Heart Disease by using Knn Algorithm

# Getting Data
url <- "http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat"
heart <- as.data.frame(read.table(file = url, header = FALSE, dec = ".", sep = " "))


#Converting type of Target variable
table(heart$V14)
heart$V14 <- as.factor(heart$V14)

# Changing Target variables values. 0 indicates 'no disease', 1 indicates 'disease exists'
table(heart$V14)
levels(heart$V14) <- c("0", "1")
table(heart$V14)

# Changing reference level of factor variable. This is important for confusion matrix
table(heart$V14)
heart$V14 <- relevel(heart$V14, ref = "1")
table(heart$V14)

# Summary statistics of heart dataset
summary(heart)

# Predictive attributes are normalized
# install.packages("clusterSim")
library(clusterSim)
heart[,-14] <-data.Normalization(heart[,-14],type="n4",normalization="column")


#Applying Knn algorithm with Euclidean Distance

# Data Partitioning
# install.packages("caret")
library(caret)
set.seed(1)
trainIndices <- createDataPartition(y = heart$V14, p = .80, list = FALSE) 
trainIndices[1:20]

# Creating training and test data set
trainset <- heart[trainIndices,]
testset <- heart[-trainIndices,]

# Checking stratified hold out 
table(heart$V14)
table(trainset$V14)
table(testset$V14)

# Separating target and predictive attributes
testPredictive <- testset[, -14] 
testTarget <- testset[[14]] 

trainPredictive <- trainset[, -14]
trainTarget <- trainset[[14]]

# Assigning value of k for KNN algorithm
k_value <- 3
# install.packages("class")
library(class)
set.seed(1)
predictions<- knn(trainPredictive, testPredictive, trainTarget, k = k_value)
#Lokking predictions results
predictions

#Performance Evaluation of Model
(result <- table(predictions, testTarget, dnn = c("Predictions", "Real Values")))

(tp <- result[1])
(fp <- result[3])
(fn <- result[2])
(tn <- result[4])

paste0("Accuracy = ",(accuracy <- (tp+tn)/sum(result)))
paste0("Error = ",(error <- 1-accuracy))
paste0("Sensitivity (TPR) = ",(TPR <- tp/(tp+fn)))
paste0("Specificity (SPC) = ",(SPC <- tn/(fp+tn)))
paste0("Precision (PPV) = ",(PPV <- tp/(tp+fp)))
paste0("Negative Predictive Value (NPV) = ",(NPV <- tn/(tn+fn)))
paste0("False Positive Rate (FPR) = ",(FPR <- fp/(fp+tn)))
paste0("False Negative Rate (FNR) = ",(FNR <- fn/(fn+tp)))
paste0("Positive Likelihood Rate (LR_p) = ",(LR_p <- TPR/FPR))
paste0("Negative Likelihood Rate (LR_n) = ",(LR_n <- FNR/SPC))
paste0("Diagnostic Odds Ratio (DOR) = ",(DOR <- LR_p/LR_n))
paste0("F_measure = ",(F_measure <- (2*PPV*TPR)/(PPV+TPR)))

# Alternative Performance Evaluation
library(caret)
result2 <- confusionMatrix(data = predictions, reference = testTarget, mode = "everything")
result2$byClass["F1"]
result2$byClass["Neg Pred Value"]
result2$overall["Accuracy"]

