#######################################
#               part one              #
#######################################

#loading the data
data <- read.csv('CarPrice_Assignment.csv')

#summary of the features
print (summary(data))



#selecting 3 features
selected = data.frame(data["enginesize"], data["horsepower"], data["enginetype"])


#plotting boxplots
png(file = "enginesize_enginetype.png")

boxplot(enginesize ~ enginetype, data = selected, xlab = "engine type",
        ylab = "engine size", main = "CarPrice dataset")

dev.off()

png(file = "horsepower_enginetype.png")

boxplot(horsepower ~ enginetype, data = selected, xlab = "engine type",
        ylab = "horsepowers", main = "CarPrice dataset")

dev.off()

png(file = "horsepower_enginesize.png")

boxplot(horsepower ~ enginesize, data = selected, xlab = "engine size",
        ylab = "horsepowers", main = "CarPrice dataset")

dev.off()

png(file = "enginesize_horsepower.png")

boxplot(enginesize~ horsepower , data = selected, xlab = "horsepower",
        ylab = "engine size", main = "CarPrice dataset")

dev.off()




#filling in the missing values

#replacing missing strings with NA
data[data==""]<-NA

#finding columns with at least one missing value
missing_cols <- colnames(data)[apply(data, 2, anyNA)]
print(missing_cols)

#Carbody and cylendernumber have string value. Hence, we replace missing 
#values with the mode of these columns, which has the maximum likelihood.

data[is.na(data[,"carbody"]), "carbody"] <- mode(data[,"carbody"])

data[is.na(data[,"cylindernumber"]), "cylindernumber"] <- mode(data[,"cylindernumber"])

#For curbweight and boreratio which have real number values, we replace missing
#values with the mean of these columns, which is an unbiased estimator for 
#the missing values.

data[is.na(data[,"curbweight"]), "curbweight"] <- mean(data[,"curbweight"], na.rm = TRUE)

data[is.na(data[,"boreratio"]), "boreratio"] <- mean(data[,"boreratio"], na.rm = TRUE)

#check if all the missing values have been replaces
missing_cols <- colnames(data)[apply(data, 2, anyNA)]
print(missing_cols)





#plotting the correlation map 
#here we plot the correlation map for two cases: first if we ignore categorical data
#second, if we replace categorical data with dummy values

#correlation map for numeric columns
numeric_cols <- unlist(lapply(data, is.numeric))
numeric_cols["X"]<-FALSE
numeric_cols["car_ID"]<-FALSE
numeric_cols["symboling"]<-FALSE

corr_matrix <- cor(data[,numeric_cols])
print (round(corr_matrix,2))

#make it a bit more beautiful
#install.packages("corrplot")
library(ggcorrplot)

ggcorrplot(corr_matrix, lab=TRUE)




#t-test for correlations

#engine size - price: Since the p_value is small the null hypothesis 
#that the two variables are not correlated is ruled out.
res <- cor.test(data$enginesize, data$price, method = "pearson")
print (res)

#city mpg - price: Since the p_value is small the null hypothesis 
#that the two variables are not correlated is ruled out.
res <- cor.test(data$citympg, data$price, method = "pearson")
print (res)

#engine size - price: Since the p_value is small the null hypothesis 
#that the two variables are not correlated is ruled out.
res <- cor.test(data$enginesize, data$curbweight, method = "pearson")
print (res)

#engine size - price: Since the p_value is greater than 0.05, the null hypothesis 
#that the two variables are not correlated is valid. Hence there is no 
#specific correlation between price and car height.
res <- cor.test(data$carheight, data$price, method = "pearson")
print (res)





#making dummy variables
#install.packages("fastDummies")
library(fastDummies)
data <- dummy_cols(subset(data, select = -c(CarName)))
print(head(data))
print (ncol(data))

#removing categorical values
data <- subset(data, select = -c(fueltype, aspiration, doornumber, carbody, drivewheel, enginelocation, enginetype, cylindernumber, fuelsystem))





#splitting the data to train and test sets
train_size <- floor(0.8 * nrow(data))

set.seed(402)
train_indices <- sample(seq_len(nrow(data)), size = train_size, replace=FALSE)

train <- data[train_indices, ]
test <- data[-train_indices, ]

print (nrow(train))
print (nrow(test))




#######################################
#               part two              #
#######################################

#building the multiple regression model
cleaned_train_data <- subset(train, select = -c(X,car_ID,symboling))
cleaned_test_data <- subset(test, select = -c(X,car_ID,symboling))
model <- lm(price ~., data = cleaned_train_data)

print(summary(model))
print(alias(model))





#reporting metrics on train
preds <- predict(model,cleaned_train_data)

#rss
rss <- sum((preds-cleaned_train_data$price)^2)
print (rss)

#tss
tss <- sum((cleaned_train_data$price-mean(cleaned_train_data$price))^2)
print (tss)

#mse
mse <- mean((preds-cleaned_train_data$price)^2)
print (mse)

#R^2 = 1-(RSS/TSS)
R2 <- 1-(rss/tss)
print (R2)

#adjusted R^2 = 1-(1-R^2)((n-k)/(n-1))
AR2 <- 1-(1-R2)*((nrow(cleaned_train_data)-1)/(nrow(cleaned_train_data)-43))
print(AR2)



#reporting metrics on test
preds <- predict(model,cleaned_test_data)

#rss
rss <- sum((preds-cleaned_test_data$price)^2)
print (rss)

#tss
tss <- sum((cleaned_test_data$price-mean(cleaned_test_data$price))^2)
print (tss)

#mse
mse <- mean((preds-cleaned_test_data$price)^2)
print (mse)

#R^2 = 1-(RSS/TSS)
R2 <- 1-(rss/tss)
print (R2)

#adjusted R^2 = 1-(1-R^2)((n-k)/(n-1))
AR2 <- 1-(1-R2)*((nrow(cleaned_test_data)-1)/(nrow(cleaned_test_data)-43))
print(AR2)






#plotting coefficients
#install.packages("coefplot")
library(coefplot)
coefplot(model, parm = -1)


#normalizing the data and plotting coefficients of the normalized data
#install.packages("caret")
library(caret)

#here we scale each feature to range [0,1]
normalized_cleaned_train_data <- predict(preProcess(cleaned_train_data, method=c("range")), cleaned_train_data)
normalized_cleaned_test_data <-  predict(preProcess(cleaned_test_data, method=c("range")), cleaned_test_data)

#fitting a new model on normalized data
normalized_model = lm(price ~., data = normalized_cleaned_train_data)

print(summary(normalized_model))
coefplot(normalized_model, parm = -1)



#######################################
#               part four             #
#######################################
# here we use lasso, which is method to obtain a more interpretable model
# because of the l1 penalty in lasso, some of the coefficients become zero
# during the training, hence the process of feature selection is done automatically
# during the training. 

#install.packages("glmnet")
library(glmnet)


#l1 coefficient is a hyperparameter, and the best value for l1 coefficient
# must be obtained with hyperparameter fine-tuning with cross validation(i.e. testing various values for
#l1 coefficient on validation data (using k-fold cross validation) and selecting the best one).

l1_coefs <- 10^seq(2, -3, by = -.1)

x_train <- as.matrix(subset(normalized_cleaned_train_data, select = -c(price)))
y_train <- normalized_cleaned_train_data$price
x_test <- as.matrix(subset(normalized_cleaned_test_data, select = -c(price)))
y_test <- normalized_cleaned_test_data$price

lasso_models <- cv.glmnet(x_train, y_train, alpha = 1, lambda = l1_coefs, standardize = TRUE, nfolds = 5)

best_coef <- lasso_models$lambda.min

print (best_coef)

lasso_model <- glmnet(x_train, y_train, alpha = 1, lambda = best_coef, standardize = TRUE)

print(predict(lasso_model, type="coef"))

preds <- predict(lasso_model, s = best_coef, newx = x_train)

#rss
rss <- sum((preds-y_train)^2)
print (rss)

#tss
tss <- sum((y_train-mean(y_train))^2)
print (tss)

#mse
mse <- mean((preds-y_train)^2)
print (mse)

#R^2 = 1-(RSS/TSS)
R2 <- 1-(rss/tss)
print (R2)


preds <- predict(lasso_model, s = best_coef, newx = x_test)

#rss
rss <- sum((preds-y_test)^2)
print (rss)

#tss
tss <- sum((y_train-mean(y_test))^2)
print (tss)

#mse
mse <- mean((preds-y_test)^2)
print (mse)

#R^2 = 1-(RSS/TSS)
R2 <- 1-(rss/tss)
print (R2)






