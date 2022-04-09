## ----Install Packages, include=FALSE---------------------------------------------------------------------------------
##Installing Packages
# List of packages for session
.packages = c("tidyverse",       #tidy alvvays and forever!
              "corrplot",        #correlation plots
              "cowplot",         #solve x-axis misalignment when plotting, and better-looking defaults for ggplots
              "gridExtra",       #combine plots
              "knitr",           #report output
              "kableExtra",      #nice tables
              "lubridate",       #date math!
              "reshape2",        #acast to create matrix
              "scales",          #get rid of scientific notation in charts
              "splitstackshape",  #explode pipe-delimited data to one-hot encoded dummy variables
              "dplyr",
              "tm",
              "tmap",
              "wordcloud",
              "knitr",
              "tinytex",
              "kableExtra",
              "tidyr",
              "stringr",
              "ggplot2",
              "gbm",
              "caret",
              "xgboost",
              "e1071",
              "class",
              "ROCR",
              "randomForest",
              "PRROC",
              "reshape2",
              "caTools",
              "Rtsne",
              "data.table"
             
              )


# Install CRAN packages (if not already installed)
.inst <- .packages %in% installed.packages()
if(length(.packages[!.inst]) > 0) install.packages(.packages[!.inst])
# Load packages into session 
lapply(.packages, require, character.only=TRUE)
tinytex::install_tinytex()




# Customize knitr output
#Set Thousands Separator for inline output
knitr::knit_hooks$set(inline = function(x) { if(!is.numeric(x)){ x }else{ prettyNum(round(x,2), big.mark=",") } })
#we've already set the graphic device to "png" in the RMD options. the default device for pdfs draws every point of a scatterplot, creatinvg *very* big files.
#But png is not as crisp, so we will set a higher resolution for pdf output of plots. 
knitr::opts_chunk$set(dpi=150)
#Create Kable wrapper function for thousands separator in table output, and nice formating with kableExtra
niceKable = function(...) {
  knitr::kable(..., format.args = list(decimal.mark = '.', big.mark = ",")) %>% kable_styling()
}

knitr::opts_chunk$set(echo = TRUE)


# loading the data
if (!(file.exists('creditcard.csv'))) {
  download.file("http://ngocanhparis.online/creditcard.csv",'creditcard.csv')
}
df = read.csv('creditcard.csv')

#Overview Dataset
tribble(
  ~"Dataset",     ~"Number of Rows",    ~"Number of Columns",
  #--             |--                   |----
  "Credit card",   nrow(df),            ncol(df),
  
)%>%niceKable


## Check missing data
sapply(df, function(x) sum(is.na(x)))%>% niceKable


#Check for class imbalance
df %>%
  group_by(Class) %>% 
  summarise(Count = n()) %>% niceKable


## Checking Class imbalance

common_theme <- theme(plot.title = element_text(hjust = 0.5, face = "bold"))
ggplot(data = df, aes(x = factor(Class), y = prop.table(stat(count)), 
                             fill = factor(Class),
                             label = scales::percent(prop.table(stat(count))))) +
  scale_fill_brewer(palette = "Set2") +
  geom_bar(position = "dodge") +
  geom_text(stat = 'count',
            position = position_dodge(.9),
            vjust = -0.5,
            size = 3) +
  scale_x_discrete(labels = c("no fraud", "fraud")) +
  scale_y_continuous(labels = scales::percent) +
  labs(x = 'Class', y = 'Percentage') + 
  ggtitle ("Distribution of Class Variable") +
  common_theme


#Distribution of Time of Transaction by Class
df %>%
  ggplot(aes(x = Time, fill = factor(Class))) + geom_histogram(bins = 100)+
  labs(x = 'Time in Seconds Since First Transaction', y = 'No. of Transactions') +
  ggtitle('Distribution of Time of Transaction by Class') +
  scale_fill_brewer(palette = "Set2") +
  facet_grid(Class ~ ., scales = 'free_y') + common_theme


#Distribution of Transaction Amount by Class
ggplot(df,aes(x = factor(Class), y =  Amount)) +
  geom_boxplot()+
  labs(x= 'Class (non-Fraud vs Fraud)', y = 'Amount (Euros)') +
  ggtitle("Distribution of Transaction Amount by Class") +
  common_theme



#Frauds Amounts Distributions
df[df$Class == 1,] %>%
  ggplot(aes(Amount))+
  geom_histogram(col = "black", fill ="darkseagreen3",binwidth = 40)+
  labs(x ='Amount', y ='Frequency')+
  ggtitle('Frauds Amounts Distributions')


#Correlation of Fraud Dataset Variables
correlations <- cor(df[,], method='spearman')
round(correlations, 2)
##title <- "Correlation of Fraud Dataset Variables"
corrplot(correlations, number.cex = .9, type = "full",
              method = "color", tl.cex=0.8,tl.col = "black")


#Remove the “Time” column and Change ‘Class’ variable to factor from the dataset
# Set seed for reproducibility
set.seed(1234)
# Remove the "Time" column from the dataset
df <- df %>% select(-Time)
#Change 'Class' variable to factor
df$Class <- as.factor(df$Class)

# Split the dataset into train, test and cross validation dataset
train_index = createDataPartition(y = df$Class,
                                  p = .6,
                                  list = F)
train <- df[train_index,]
test_cv <-df[-train_index,]

test_index = createDataPartition(y = test_cv$Class,
                                  p = .5,
                                  list = F)

test <- test_cv[test_index,]
cv <- test_cv[-test_index,]

rm(train_index, test_index, test_cv)


##Naive Algorithm
# Set seed 1234 for reproducibility
set.seed(1234)
# Build the model with Class as target and all other variables as predictors
naive_model <- naiveBayes(Class ~ ., data = train, laplace=1)
#Prediction with training and test data
predictions <- predict(naive_model, newdata=test)
# Compute the AUC and AUCPR for the Naive Model
pred <- prediction(as.numeric(predictions), test$Class)

auc_naive <- performance(pred, "auc")
auc_p_naive <- performance(pred, 'sens', 'spec')
aucpr_p_naive <- performance(pred, "prec", "rec")

aucpr_naive <- pr.curve(
  scores.class0 = predictions[test$Class == 1], 
  scores.class1 = predictions[test$Class == 0],
  curve = T,  
  dg.compute = T
)

# Make the relative plot of Naive Algorithm
plot(aucpr_naive)
plot(auc_p_naive, main=paste("AUC:", auc_naive@y.values[[1]]))
plot(aucpr_p_naive, main=paste("AUCPR:", aucpr_naive$auc.integral))


# Create a dataframe 'results' that contains all metrics 
# obtained by the trained models

results <- data.frame(
  Model = "Naive Bayes", 
  AUC = auc_naive@y.values[[1]],
  AUCPR = aucpr_naive$auc.integral
)

# Show results of Naive Algorithm on a table

results %>% 
  kable() %>%
  kable_styling(
    bootstrap_options = 
      c("striped", "hover", "condensed", "responsive"),
      position = "center",
      font_size = 10,
      full_width = FALSE
)




## # Build a KNN Model with Class as Target and all other variables as predictors. k is set to 5
# Set seed 1234 for reproducibility
set.seed(1234)

knn_model <- knn(train[,-30], test[,-30], train$Class, k=5, prob = TRUE)

# Compute the AUC and AUCPR for the KNN Model

pred <- prediction(
  as.numeric(as.character(knn_model)), as.numeric(as.character(test$Class))
)

auc_knn <- performance(pred, "auc")

auc_p_knn <- performance(pred, 'sens', 'spec')
aucpr_p_knn <- performance(pred, "prec", "rec")

aucpr_knn <- pr.curve(
  scores.class0 = knn_model[test$Class == 1], 
  scores.class1 = knn_model[test$Class == 0],
  curve = T,  
  dg.compute = T
)

# Make the relative plot for the KNN

plot(aucpr_knn)
plot(auc_p_knn, main=paste("AUC:", auc_knn@y.values[[1]]))
plot(aucpr_p_knn, main=paste("AUCPR:", aucpr_knn$auc.integral))

# Adding the respective metrics to the results dataset

results <- results %>% add_row(
  Model = "K-Nearest Neighbors k=5", 
  AUC = auc_knn@y.values[[1]],
  AUCPR = aucpr_knn$auc.integral
)

# Show results of KNN on a table

results %>% niceKable



## ----RandomForest
# Set seed 1234 for reproducibility

set.seed(1234)

# Build a Random Forest Model with Class as Target and all other
# variables as predictors. The number of trees is set to 500

rf_model <- randomForest(Class ~ ., data = train, ntree = 500)

# Get the feature importance of RandomForest

feature_imp_rf <- data.frame(importance(rf_model))

# Make predictions based on Random Forest model

predictions <- predict(rf_model, newdata=test)

# Compute the AUC and AUPCR of Random Forest

pred <- prediction(
  as.numeric(as.character(predictions)),                                 as.numeric(as.character(test$Class))
)

auc_val_rf <- performance(pred, "auc")

auc_plot_rf <- performance(pred, 'sens', 'spec')

aucpr_plot_rf <- performance(pred, "prec", "rec", curve = T,  dg.compute = T)

aucpr_val_rf <- pr.curve(scores.class0 = predictions[test$Class == 1], scores.class1 = predictions[test$Class == 0],curve = T,  dg.compute = T)

# make the relative plot of Random Forest

plot(auc_plot_rf, main=paste("AUC:", auc_val_rf@y.values[[1]]))
plot(aucpr_plot_rf, main=paste("AUCPR:", aucpr_val_rf$auc.integral))
plot(aucpr_val_rf)

# Adding the respective metrics to the results dataset

results <- results %>% add_row(
  Model = "Random Forest",
  AUC = auc_val_rf@y.values[[1]],
  AUCPR = aucpr_val_rf$auc.integral)

# Show results of Random Forest on a table

results %>% niceKable

# Show feature importance on a table - Random Forest

feature_imp_rf %>% niceKable


## ----XGBoost, echo=FALSE, include=TRUE-------------------------------------------------------------------------------
# Set seet 1234 for reproducibility

set.seed(1234)

# Prepare the training dataset for XGBoost model

xgb_train <- xgb.DMatrix(
  as.matrix(train[, colnames(train) != "Class"]), 
  label = as.numeric(as.character(train$Class))
)

# Prepare the test dataset for XGBoost model

xgb_test <- xgb.DMatrix(
  as.matrix(test[, colnames(test) != "Class"]), 
  label = as.numeric(as.character(test$Class))
)

# Prepare the cv dataset for XGBoost model

xgb_cv <- xgb.DMatrix(
  as.matrix(cv[, colnames(cv) != "Class"]), 
  label = as.numeric(as.character(cv$Class))
)

# Prepare the parameters list for XGBoost model

xgb_params <- list(
  objective = "binary:logistic", 
  eta = 0.1, 
  max.depth = 3, 
  nthread = 6, 
  eval_metric = "aucpr"
)

# Train the XGBoost Model

xgb_model <- xgb.train(
  data = xgb_train, 
  params = xgb_params, 
  watchlist = list(test = xgb_test, cv = xgb_cv), 
  nrounds = 500, 
  early_stopping_rounds = 40,
  verbosity = 0,
  print_every_n = 100,
  silent = T,

)

# Get feature importance on XGBoost model
feature_imp_xgb <- xgb.importance(model = xgb_model)
xgb.plot.importance(feature_imp_xgb, rel_to_first = TRUE, xlab = "Relative importance")

# Make predictions based on  XGBoost model model

predictions = predict(
  xgb_model, 
  newdata = as.matrix(test[, colnames(test) != "Class"]), 
  ntreelimit = xgb_model$bestInd
)

# Compute the AUC and AUPCR on  XGBoost model

pred <- prediction(
  as.numeric(as.character(predictions)),                                 as.numeric(as.character(test$Class))
)

auc_val_xgb <- performance(pred, "auc")

auc_plot_xgb <- performance(pred, 'sens', 'spec')
aucpr_plot_xgb <- performance(pred, "prec", "rec")

aucpr_val_xgb <- pr.curve(
  scores.class0 = predictions[test$Class == 1], 
  scores.class1 = predictions[test$Class == 0],
  curve = T,  
  dg.compute = T
)

# Make the relative plot on  XGBoost model

plot(auc_plot_xgb, main=paste("AUC:", auc_val_xgb@y.values[[1]]))
plot(aucpr_plot_xgb, main=paste("AUCPR:", aucpr_val_xgb$auc.integral))
plot(aucpr_val_xgb)

# Adding the respective metrics to the results dataset

results <- results %>% add_row(
  Model = "XGBoost",
  AUC = auc_val_xgb@y.values[[1]],
  AUCPR = aucpr_val_xgb$auc.integral)

# Show results on a table of XGBoost model

results %>% niceKable

# Show feature importance on a table

feature_imp_xgb %>% niceKable


# Shows the results of these model
results %>% niceKable

