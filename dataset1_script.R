#Library to be used----
library(tidyverse)
library(tidymodels)
library(tidytext)
library(SnowballC)
library(topicmodels)
library(DataExplorer)
library(themis)
library(vip)

#Read the data: data1 training, validation and testing data----
data1_training_raw <- read_csv("00 data/data1/training.csv")

data1_validation_raw <- read_csv("00 data/data1/validation.csv")

data1_testing_raw <- read_csv("00 data/data1/test.csv")

#Data cleansing & wrangling----
##Handle NA in the dataset----
###Check whether the column in training dataset has empty value----
data1_training_raw %>% summarise_all(~sum(is.na(.)) / length(.) * 100) ####Each column contains at least 1365/4.35% empty value which is not significant

###Drop the rows that has at least one empty value----
data1_training_raw <- drop_na(data1_training_raw) 

###Count occurrence and relative frequency of each unique value in S column----
data1_training_raw %>% 
    count(S) %>% 
    mutate(
        relative_frequency = (n / sum(n)) * 100
    ) %>% 
    arrange(desc(relative_frequency)) ####There are over 2000 unique values in S column, and the top 10 values only account for 10% of the total data


##Convert TO column from string type to factor type----
data1_training_raw <- data1_training_raw %>% 
    mutate(
        TO = as_factor(TO)
    )

##Convert class label from number to factor type----
data1_training_raw <- data1_training_raw %>% 
    mutate(
        `class label` = as_factor(`class label`)
    )

##Convert text data from T1 and T2 to numeric value via feature extraction----
data1_training_text_processed <- data1_training_raw %>% 
    
    ###Combine T1 and T2 columns into one column called T1_T2----
    mutate(
        T1_T2 = paste(T1, T2, sep = " ")
    ) %>% 
    
    ###Remove punctuation and anything that is not English letter and space----
    mutate(
        T1_T2 = str_replace_all(T1_T2, "[^a-zA-Z\\s]", "")
    ) %>% 
    
    ###Change all to lower case in T1_T2----
    mutate(
        T1_T2 = str_to_lower(T1_T2)
    ) %>% 
    
    ###Remove the stopwords----
    unnest_tokens(
        input = T1_T2,
        output = word,
        token = "words",
        drop = FALSE
    ) %>% 
    anti_join(stop_words) %>% 
    
    ###Stemming the words via SnowballC package----
    mutate(
        word = wordStem(word, language = "en")
    )
  

###Create a Document-Term Matrix (DTM)----
dtm <- data1_training_text_processed %>%
    count(id, word) %>%
    cast_dtm(document = id, term = word, value = n)

###Perform topic modelling----
###Finding optimal topic numbers----
result <- ldatuning::FindTopicsNumber(
    dtm,
    topics = seq(from = 2, to = 50, by = 1),
    metrics = c("CaoJuan2009",  "Deveaud2014", "Griffiths2004", "Arun2010"),
    method = "Gibbs",
    control = list(seed = 77),
    verbose = TRUE
)

ldatuning::FindTopicsNumber_plot(result)

###Perform Latent Dirichlet Allocation (LDA) for k topics----
k_topics = 50
lda_model <- LDA(dtm, k = k_topics) 

###Get the most important words in each topic----
terms(lda_model, 5)

###Get the distribution of topics in each document----
topic_distribution <- posterior(lda_model)$topics

###Convert the topic distribution to a dataframe----
topic_distribution_df <- as.data.frame(topic_distribution)

###Add the topic distribution dataframe to the original dataframe----
data1_training_cleaned <- cbind(data1_training_raw, topic_distribution_df)

#Data exploration for training dataset only----
##Create EDA report for the training dataset to see if data manipulation is needed----
create_report(data1_training_cleaned) 

#Perform the same data cleansing steps for validation and testing dataset----
##Convert TO column from string type to factor type----
data1_validation_raw <- data1_validation_raw %>% 
    mutate(
        TO = as_factor(TO)
    )

data1_testing_raw <- data1_testing_raw %>% 
    mutate(
        TO = as_factor(TO)
    )

##Convert class label from number to factor type----
data1_validation_raw <- data1_validation_raw %>% 
    mutate(
        `class label` = as_factor(`class label`)
    )

##Convert text data from T1 and T2 to numeric value via feature extraction----
###Create a function to perform the same steps----
process_text_data <- function(df) {
    data_processed <- df %>% 
        mutate(T1_T2 = paste(T1, T2, sep = " ")) %>% 
        mutate(T1_T2 = str_replace_all(T1_T2, "[^a-zA-Z\\s]", "")) %>% 
        mutate(T1_T2 = str_to_lower(T1_T2)) %>% 
        unnest_tokens(input = T1_T2, output = word, token = "words", drop = FALSE) %>% 
        anti_join(stop_words) %>% 
        mutate(word = wordStem(word, language = "en"))
    
    dtm <- data_processed %>%
        count(id, word) %>%
        cast_dtm(document = id, term = word, value = n)
    
    lda_model <- LDA(dtm, k = k_topics) 
    
    topic_distribution <- posterior(lda_model)$topics
    topic_distribution_df <- as.data.frame(topic_distribution)
    
    text_processed_final <- cbind(df, topic_distribution_df)
    
    return(text_processed_final)
}

###Perform the function on validation and testing dataset----
data1_validation_cleaned <- process_text_data(data1_validation_raw)

data1_testing_cleaned <- process_text_data(data1_testing_raw)

#Build machine learning model----
##Feature engineering for training dataset via recipe----
data1_recipe <- recipe(`class label` ~ ., data = data1_training_cleaned) %>%
    
    ###Remove not-needed columns----
    step_rm(id) %>% 
    step_rm(T1) %>% 
    step_rm(T2) %>% 
    
    ###Value are grouped into "other" if its relative frequency is lower than 1%----
    step_other(S, threshold = 0.01) %>%
    
    ###Converts characters or factors (i.e, nominal variables) into one or more numeric binary model terms for the levels of the original data----
    step_dummy(all_nominal(), -all_outcomes()) %>% 
    
    ###Removes indicator variables that only contain a single unique value (e.g. all zeros)----
    step_zv(all_predictors()) %>% 
    
    ###Reduce class imbalance effect via SMOTE algorithm----
    step_smote(`class label`)

##Xgboost is built on the training dataset----
xgb_mod <- 
    boost_tree(
      trees = tune(),
      tree_depth = tune(), min_n = tune(),
      loss_reduction = tune(),                     ### first three: model complexity
      sample_size = tune(), mtry = tune(),         ### randomness
      learn_rate = tune()                          ### step size
    ) %>%  
    set_engine("xgboost", tree_method = "hist", device = "cuda") %>% 
    set_mode("classification")


##Create a workflow for the model to bundle our parsnip model (rf_mod) with our recipe (data1_recipe)----
class_label_wflow <- 
    workflow() %>% 
    add_model(xgb_mod) %>% 
    add_recipe(data1_recipe)

##Create a set of 10-folded cross-validation resamples to use for tuning----
set.seed(55555)
mod_folds <- vfold_cv(data1_training_cleaned, v = 10, strata = `class label`)

##Tune the model via grid search----
macro_f1 <- metric_tweak("macro_f1", f_meas, estimator = "macro")

micro_f1 <- metric_tweak("micro_f1", f_meas, estimator = "micro")

f1_metrics_set <- metric_set(macro_f1, micro_f1)

xgb_para_set <- class_label_wflow %>%
  extract_parameter_set_dials() %>%
  update(tree_depth = tree_depth(range = c(1, 50)), min_n = min_n(range = c(1, 100))) %>% 
  finalize(select(data1_training_cleaned, -`class label`))

xgb_grid_latin_hypercube <- xgb_para_set %>% grid_latin_hypercube(
  size = 50
)

start_time_display <- Sys.time()

set.seed(55555)
tune_res <- tune_grid(
    class_label_wflow,
    resamples = mod_folds,
    metrics = metric_set(macro_f1, micro_f1),
    grid = xgb_grid_latin_hypercube,
    control = control_grid(save_pred = TRUE)
)

end_time_display <- Sys.time()
print(paste("Mode training time", end_time_display - start_time_display)) #[1] "Mode training time 10.413169110616"

###Create confusion matrix and then shown in percentage---
confusion_matrix <- tune_res %>%
    collect_predictions() %>%
    conf_mat(truth = `class label`, estimate = .pred_class) 


###Check the performance of the model----
tune_res %>%
    collect_metrics() %>%
    filter(.metric == "macro_f1") %>%
    select(mean, min_n, mtry, trees, tree_depth, loss_reduction, sample_size, learn_rate) %>%
    pivot_longer(min_n:learn_rate,
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
    select(mean, min_n, mtry,trees,tree_depth, loss_reduction, sample_size, learn_rate) %>%
    pivot_longer(min_n:learn_rate,
                 values_to = "value",
                 names_to = "parameter"
    ) %>%
    ggplot(aes(value, mean, color = parameter)) +
    geom_point(show.legend = FALSE) +
    facet_wrap(~parameter, scales = "free_x") +
    labs(x = NULL, y = "micro_f1")

###Show best model via macro_f1 and micro_f1----
show_best(tune_res, "macro_f1")

show_best(tune_res, "micro_f1")

###Select the best model and then predict on validation dataset----
best_model <- select_best(tune_res, "micro_f1")

final_model_wf <- finalize_workflow(
  class_label_wflow,
  best_model
)

####Feature importance----
final_model_wf %>%
  fit(data = data1_training_cleaned) %>%
  pull_workflow_fit() %>%
  vip(geom = "point")

###Evaluate the model on validation dataset----
final_model <- finalize_model(
  xgb_mod,
  best_model
)

final_wf <- workflow() %>%
  add_recipe(data1_recipe) %>%
  add_model(final_model)

final_wf_fit <- fit(final_wf, data = data1_training_cleaned)
final_wf_predict <- predict(final_wf_fit, new_data = data1_validation_cleaned)

validation_f1 <- final_wf_predict %>% 
  bind_cols(data1_validation_cleaned) %>% 
  f1_metrics_set(truth = `class label`, estimate = .pred_class)

###Predict on test dataset----
final_wf_predict_test <- predict(final_wf_fit, new_data = data1_testing_cleaned)

testing_prediction_result <- final_wf_predict_test %>% 
  bind_cols(data1_testing_cleaned) %>% 
  select(-`class label`) %>% 
  rename(`class label` = .pred_class)

write_csv(testing_prediction_result, "testing_prediction_result.csv")











