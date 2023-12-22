#Library to be used----
library(tidyverse)
library(tidymodels)
library(tidytext)
library(SnowballC)
library(topicmodels)
library(DataExplorer)
library(themis)
library(mice)
library(vip)

#Read the data: data1 training, validation and testing data----
data2_training_raw <- read_csv("00 data/data2/training.csv")

data2_validation_raw <- read_csv("00 data/data2/validation.csv")

data2_testing_raw <- read_csv("00 data/data2/test.csv")

#Data cleansing & wrangling----
##Convert chr type to number type----
data2_training_raw <- data2_training_raw %>% 
  
  ###replace ? by NA
  mutate(across(where(is.character), ~na_if(., "?"))) %>% 
  
  ###convert chr to number
  mutate(across(where(is.character), as.numeric)) 

#Data exploration for training dataset only----
##Create EDA report for the training dataset----
create_report(data2_training_raw)

##Handle NA in the dataset----
###Assume the missing data type is either MCAR or MAR to conduct predictive mean matching----
data2_training_raw <- data2_training_raw %>% 
  rename(class_label = `class label`) 

data2_training_raw.imp <- mice(data2_training_raw, m = 5, method = 'pmm',seed=500)

summary(data2_training_raw.imp)

####Obtain first imputed dataset
data2_training_clean <- complete(data2_training_raw.imp, 1)

#Perform the same data cleansing steps for validation and testing dataset----
##Create a function to perform the same steps----
convert_data <- function(data) {
  
  data <- data %>% 
    mutate(across(where(is.character), ~na_if(., "?"))) %>% 
    mutate(across(where(is.character), as.numeric)) %>% 
    rename(class_label = `class label`)
  
  data.imp <- mice(data, m = 5, method = 'pmm', seed=500)
  
  data_clean <- complete(data.imp, 1)
  
  return(data_clean)
}

data2_validation_clean <- convert_data(data2_validation_raw)

data2_testing_clean <- convert_data(data2_testing_raw)

#Build machine learning model----
##Feature engineering for training dataset via recipe----
data2_recipe <- recipe(class_label ~ ., data = data2_training_clean) %>%
    
    ###Remove not-needed columns----
    step_rm(id) %>% 
    
    ###Convert class label to factor----
    step_num2factor(class_label, 
                    levels = c("1", "2", "3", "4", "5"),
                    skip = TRUE) %>% 
  
    ###Normalize the predictors----
    step_YeoJohnson(all_predictors()) %>% 
    step_normalize(all_predictors()) %>% 
    
    ###Removes indicator variables that only contain a single unique value (e.g. all zeros)----
    step_zv(all_predictors()) 
    
##Random Forrest is built on the training dataset----
rf_mod <-
    rand_forest(

        ###Set the number of trees to grow via gird search in later stage----
        trees = 1000,

        ###Set the number of variables randomly sampled as candidates at each split via gird search in later stage----
        mtry = tune(),

        ###Set the minimum number of observations in the terminal nodes via gird search in later stage----
        min_n = tune()

    ) %>%
    set_engine("ranger") %>%
    set_mode("classification")

##Create a workflow for the model to bundle our parsnip model (rf_mod) with our recipe (data2_recipe)----
class_label_wflow <- 
    workflow() %>% 
    add_model(rf_mod) %>% 
    add_recipe(data2_recipe)

##Create a set of 10-folded cross-validation resamples to use for tuning----
set.seed(55555)
mod_folds <- vfold_cv(data2_training_clean, v = 10, strata = class_label)

##Tune the model via grid search----
macro_f1 <- metric_tweak("macro_f1", f_meas, estimator = "macro")

micro_f1 <- metric_tweak("micro_f1", f_meas, estimator = "micro")

f1_metrics_set <- metric_set(macro_f1, micro_f1)

start_time_display <- Sys.time()

doParallel::registerDoParallel()

set.seed(55555)
tune_res <- tune_grid(
    class_label_wflow,
    resamples = mod_folds,
    metrics = metric_set(macro_f1, micro_f1),
    grid = 20,
    control = control_grid(save_pred = TRUE)
)

doParallel::stopImplicitCluster()

end_time_display <- Sys.time()
print(paste("Mode training time", end_time_display - start_time_display)) ####"Mode training time 5.5hr"

tune_res %>%
    collect_metrics() %>%
    filter(.metric == "macro_f1") %>%
    select(mean, min_n, mtry) %>%
    pivot_longer(min_n:mtry,
                 values_to = "value",
                 names_to = "parameter"
    ) %>%
    ggplot(aes(value, mean, color = parameter)) +
    geom_point(show.legend = FALSE) +
    facet_wrap(~parameter, scales = "free_x") +
    labs(x = NULL, y = "macro_f1")

tune_res %>%
    collect_metrics() %>% 
    filter(.metric == "micro_f1") %>%
    select(mean, min_n, mtry) %>%
    pivot_longer(min_n:mtry,
                 values_to = "value",
                 names_to = "parameter"
    ) %>%
    ggplot(aes(value, mean, color = parameter)) +
    geom_point(show.legend = FALSE) +
    facet_wrap(~parameter, scales = "free_x") +
    labs(x = NULL, y = "micro_f1")

###Continue to fine tune----
rf_grid <- grid_regular(
  mtry(range = c(1, 10)),
  min_n(range = c(5, 15)),
  levels = 5
)

doParallel::registerDoParallel()

set.seed(456)
regular_res <- tune_grid(
  class_label_wflow,
  resamples = mod_folds,
  metrics = metric_set(macro_f1, micro_f1),
  grid = rf_grid
)

doParallel::stopImplicitCluster()

autoplot(regular_res, metric = "macro_f1")

autoplot(regular_res, metric = "micro_f1")

###Show best model via macro_f1 and micro_f1----
show_best(regular_res, "macro_f1")

show_best(regular_res, "micro_f1")

###Select the best model and then predict on validation dataset----
best_rf <- select_best(regular_res, "macro_f1")

final_rf <- finalize_model(
  rf_mod,
  best_rf
)

####Feature importance----
final_rf %>%
  set_engine("ranger", importance = "permutation") %>%
  fit(class_label ~ .,
      data = juice(prep(data2_recipe))
  ) %>%
  vip(geom = "point")

###Evaluate the model on validation dataset----
final_wf <- workflow() %>%
  add_recipe(data2_recipe) %>%
  add_model(final_rf)

final_wf_fit <- fit(final_wf, data = data2_training_clean)
final_wf_predict <- predict(final_wf_fit, new_data = data2_validation_clean)

validation_f1 <- final_wf_predict %>% 
  bind_cols(data2_validation_clean) %>% 
  mutate(class_label = as_factor(class_label)) %>%
  f1_metrics_set(truth = class_label, estimate = .pred_class)

###Predict on test dataset----
final_wf_predict_test <- predict(final_wf_fit, new_data = data2_testing_clean)

testing_prediction_result <- final_wf_predict_test %>% 
  bind_cols(data2_testing_clean) %>% 
  select(-class_label) %>% 
  rename(`class label` = .pred_class)

write_csv(testing_prediction_result, "testing_prediction_result.csv")
