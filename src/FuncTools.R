### Functions ####

# Function to test the association between categorical variables 
var_association <- function(var, df){
  library(rcompanion)
  library(DescTools)
  
  # Creating an empty dataframe to save the results
  results = data.frame(matrix(nrow = 0, ncol = 6))
  names(results) = c('var_x', 'var_y', 'theil_U', 'theil_sym', 'cramerV', 'chisq') # 
  
  for (vary in var){
    # Accessing the values of vary in dataframe df
    y = df[[vary]]
    
    for (varx in var){
      # Adding varx to results dataframe in column var_x
      results[(length(results$var_x)+1),'var_x'] = varx
      
      # Adding vary to results dataframe in column var_y
      results[(length(results$var_y)),'var_y'] = vary
      
      # Accessing the values of varx in dataframe df
      x = df[[varx]]
      
      # Calculating uncertainty coefficients
      # column
      theil_coef = UncertCoef(x, y, direction = 'column')
      results[(length(results$theil_U)),'theil_U'] = theil_coef
      
      # symmetric
      theil_coef = UncertCoef(x, y, direction = 'symmetric')
      results[(length(results$theil_sym)),'theil_sym'] = theil_coef
      
      # Calculating Cramer's V, corrected for bias
      cramer = cramerV(x, y, bias.correct = TRUE)
      results[(length(results$cramerV)),'cramerV'] = cramer
      
      # Calculating chi2 test
      chisq = chisq.test(x, y)
      results[(length(results$chisq)),'chisq'] = chisq$p.value
    }
  }
  return(results)
}

# Plot of correlations for categorical variables
plot.correlations <- function(correlation_dataframe){
  # Import libraries
  library(ggcorrplot)
  
  # Variables that will be used
  values =  c('theil_U', 'theil_sym', 'cramerV', 'chisq') #
  # correlation_dataframe is reshaped and transformed in a matrix
  for (value_col in values){
    data <- as.matrix(cast(correlation_dataframe, var_x~var_y,value = value_col))
    print(ggcorrplot(data, type='full', lab = TRUE, title = value_col, show.legend = FALSE))
  }
}


# Function to count time from a zero mark (specified here as the index of min value in a datetime vector).
# The index is used here to perform the calculations between a specific time and the zero mark.
time_count <- function(datetime_vector){
  
  #finding the datetime index of the first click 
  index_first_datetime = which(datetime_vector == min(datetime_vector))
  result = c()
  
  #Loop in datetime vector
  for (i in 1:length(datetime_vector)){
    # calculating the continuous time
    result[i] = as.numeric(difftime(datetime_vector[i],datetime_vector[index_first_datetime], units = 'mins'))
  }
  
  return(result)
}


### Functions to group categorical variables ###

# Using download absolute frequency. This strategy worked well in Talking Data dataset, but
# has the advantages of do not control the number of groups created.
group_categories <- function(dataframe, names, var_target){
  for(var in names){
    categories <- dataframe %>%
      filter(!!sym(var_target) == 1) %>%
      group_by((!!sym(var))) %>% # !!sym transform var (string) into a symbol
      summarise(n =n()) %>%
      filter(n > 10)
    group = nrow(categories)
    dataframe[,paste(var, 'group', sep = '_')] = as.factor(sapply(dataframe[,..var],
                                                                  function(x){ifelse(x %in% categories[[var]], x, group+1)}))
  }
  return(dataframe)
}

# Creating groups by relative frequency of downloads 
# Here I considered (number of downloads in i) / (sum of downloads) in order to have significant
# percentages. Thus, it is the relative frequency of the positive responses, instead of variable
# relative frequency.

group_target_prop <- function(dataframe, variable, target){
  for(var in variable){
    test <- as.data.table(table(dataframe[[var]], dataframe[[target]]))
    test <- reshape(test, idvar = 'V1', v.names = 'N', timevar = 'V2', direction = 'wide')
    test <- test %>%
      mutate(response_rate = (N.1 / sum(N.1))*100)
    
    # Creating a new column in dataframe with values = 0
    name = paste(var, 'response', sep = '_')
    dataframe[,name] <- 0
    
    # Loop for match the response rate with respective var values
    for(i in 1:length(test[['V1']])){
      index = which(dataframe[[var]] == test[['V1']][i])
      dataframe[[name]][index] <- test[['response_rate']][i]
    }
    
    # The number of categories will be fixed to 5
    min = min(dataframe[[name]])
    max = max(dataframe[[name]])
    increment = (max - min) /5
    intervals = c(min,min+increment, min+2*increment, min+3*increment, min+4*increment, min+5*increment)
    
    dataframe[, paste(var, 'resp_group', sep ='_')] = cut(dataframe[[name]], 
                                                          breaks = intervals,
                                                          include.lowest = TRUE,
                                                          labels = c(1,2,3,4,5))
    dataframe[[name]] <- NULL
  }
  return(dataframe)
}


# Functions for evaluating models
evaluate_model <- function(model, test_set, target_name, variables = NULL, predictions = NULL){
  
  # This function evaluates models from different algorithms based on predicted classes
  # Test_set must be a dataframe, not a sparce matrix 
  
  # predictions
  if(is.null(predictions)){
    pred = predict(model, newdata=test_set[,..variables])
  }else{
    pred = as.factor(predictions)
  }
  # Confusion Matrix (caret)
  print(confusionMatrix(pred, test_set[[target_name]], positive = 'yes', mode = "prec_recall"))
  # Data for ROC and precision-recall curves
  scores = data.frame(pred, test_set[[target_name]])
  names(scores) <- c('pred','true')
  # scores.class0 = positive class, scores.class1 = negative class
  # AUC/ROC curve
  roccurve <- PRROC::roc.curve(scores.class0 = scores[scores$true == 'yes',]$pred, 
                               scores.class1 = scores[scores$true == 'no',]$pred, 
                               curve = TRUE)
  plot(roccurve)
  # AUCPR curve
  prcurve <- PRROC::pr.curve(scores.class0 = scores[scores$true == 'yes',]$pred,
                             scores.class1 = scores[scores$true == 'no',]$pred,
                             curve = TRUE)
  plot(prcurve)
}

evaluate_model_prob <- function(model, test_data, labels){
  
  # This function returns roc_auc and prc_auc based on predicted probabilities
  
  pred = predict(model, newdata = test_data, type = 'prob')
  
  pred<-prediction(pred[, 2], labels= labels)
  
  auc_roc <- performance(pred, measure = "auc")
  auc_prc <- performance(pred, measure = "aucpr")
  return(list('roc_auc' = auc_roc@y.values[[1]], 'prc_auc' = auc_prc@y.values[[1]]))
}

# Change factor values to numeric
factor_to_numeric <- function(dataset, names){
  # Disable the warning message thrown in the second if{} of this code
  defaultW <- getOption("warn")
  options(warn = -1)
  for(name in names){
    if(is.factor(dataset[[name]])){
      if(sum(is.na(as.numeric(levels(dataset[[name]])))) > 0){
        label = levels(dataset[[name]])
        dataset[,name] <- case_when(dataset[[name]] == label[1] ~ 1, 
                                    dataset[[name]] == label[2] ~ 2, 
                                    dataset[[name]] == label[3] ~ 3,
                                    dataset[[name]] == label[4] ~ 4)
      }else{
        dataset[,name] = as.numeric(levels(dataset[[name]]))[dataset[[name]]]
      }
    }
  }
  options(warn = defaultW)
  return(dataset)
}


# Train and predict results from xgboost
train_xgboost <- function(target, variables, train_dataset, test_dataset){
  require(Matrix)
  require(xgboost)
  
  if(is.data.frame(train_dataset)){
    train_dataset = as.data.table(train_dataset)
  }
  
  if(is.data.frame(test_dataset)){
    test_dataset = as.data.table(test_dataset)
  }
  
  # formula to create sparce matrix
  f = paste(target, '~.', collapse="")
  # sparce matrix
  train_sparce <- sparse.model.matrix(as.formula(f), data = train_dataset[,..variables])
  test_sparce <- sparse.model.matrix(as.formula(f), data = test_dataset[,..variables])
  
  # Output vector
  if(is.factor(train_dataset[[target]])){
    output_vector_train = as.numeric(levels(train_dataset[[target]]))[train_dataset[[target]]]
  }else{
    output_vector_train = train_dataset[[target]]
  }
  
  # Train the model 
  set.seed(1045)
  model <- xgboost(data = train_sparce, label = output_vector_train, max.depth = 6, eta = 0.2, nrounds = 100,
                 nthread = 2, objective = "binary:logistic", params = list(eval_metric = "auc"), verbose = FALSE)
  
  # Predictions. xgboost does a regression, we have to transform the pred values to a binary classification.
  predictions = as.numeric(predict(model, test_sparce) > 0.5)
  
  return(list('model' = model, 'predictions' = predictions, variable_names = test_sparce@Dimnames[[2]]))
}


## Function to obtain the best value for perc.over parameter of SMOTE function
grid_search_smote <- function(train_data, test_data, target, features, oversamples, model = 1){
  # model == 1 -> C5.0 algorithm
  # model == 2 -> xgboost
  # else -> svm
  require(DMwR)
  for(i in oversamples){
    f = paste(target, '~.', collapse="")
    smoted_data <- SMOTE(as.formula(f), data = train_data[,..features], perc.over = i, k = 5,
                         perc.under = 400, set.seed = 1045)
    var = features[features!=target]
    if(model == 1){
      print(paste(i, 'C50'))
      set.seed(1045)
      C50_smote <- C5.0(x = smoted_data[,..var] , y = smoted_data[[target]], trials = 10)
      pred = predict(C50_smote, newdata=test_data[,..var])
    }else if(model ==2){
      print(paste(i, 'xgboost'))
      xgb <- train_xgboost(target = target, variables = features, 
                           train_dataset = smoted_data, test_dataset = test)
      pred = xgb$predictions
    }else{
      print(paste(i, 'SVM'))
      model_svm_v1 <- svm(as.formula(f), 
                          data = smoted_data, 
                          type = 'C-classification', 
                          kernel = 'radial')
      pred = predict(model_svm_v1, newdata=test_data[,..var])
    }
    print(ROSE::roc.curve(predicted = pred, response = test_data[[target]], plotit = FALSE))
    print(table(pred, test[[target]]))
  }
  
}
