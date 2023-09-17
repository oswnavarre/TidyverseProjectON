library(tidyverse)
library(tidymodels)
library(tm)
library(tidytext)
library(wordcloud)
library(doParallel)
library(parallel)
library(vip)
library(skimr)

detectCores()
doParallel::registerDoParallel(cores=8)
training <- read.csv("data_complaints_train.csv")
testing <- read.csv("data_complaints_test.csv")

training <- training |>
  rename(Complaint = Consumer.complaint.narrative) |>
  select(Product, Complaint) |>
  mutate(Product = factor(Product))

testing <- testing |>
  rename(Complaint = Consumer.complaint.narrative) |>
  select(Complaint)


training_clean <- training |> 
  mutate(Complaint=gsub("[XX]+", "", Complaint)) |> 
  mutate(Complaint=str_to_lower(Complaint)) |> 
  mutate(Complaint=gsub("[0-9]", "", Complaint)) |> 
  mutate(Complaint=removePunctuation(Complaint)) |> 
  mutate(Complaint=gsub("\n", "", Complaint)) |> 
  mutate(Complaint=gsub("\t", "", Complaint)) |> 
  mutate(Complaint=stripWhitespace(Complaint))

training_complaints <- Corpus(VectorSource(training_clean$Complaint)) |> 
  tm_map(removeWords, stopwords())

training_docterm <- DocumentTermMatrix(training_complaints)
inspect(training_docterm)

limitfreq <- findFreqTerms(training_docterm, lowfreq=1500)

training_docterm <- DocumentTermMatrix(training_complaints, list(dictionary=limitfreq)) 
inspect(training_docterm)

training_docterm <- removeSparseTerms(training_docterm, 0.95) 
inspect(training_docterm)

training_docterm_df <- training_docterm |>
  as.matrix() |>
  as.data.frame()

training_docterm_df  |>
  summarise_all(sum)

list_terms <- sort(names(training_docterm_df))

unnecessary_terms <- c("able","ago","already","also","another",
                       "anything","asked","back","believe","call","called",
                      "calling","can","didnt","done","dont","either","even","get",
                      "going","got","however","just","later","like","needed",
                      "never","now","per","please","put","said","said","saying","take","today","told",
                      "took","used","want","way","went","will","without")

training_complaints<-tm_map(training_complaints, removeWords, unnecessary_terms)

training_docterm <- DocumentTermMatrix(training_complaints, list(dictionary=limitfreq))
inspect(training_docterm)

training_docterm <- removeSparseTerms(training_docterm, 0.95)
inspect(training_docterm)

training_docterm_df <- training_docterm |>
  as.matrix() |>
  as.data.frame()

testing_clean <- testing |> 
  mutate(Complaint = gsub("[XX]+", "", Complaint)) |> 
  mutate(Complaint = str_to_lower(Complaint)) |> 
  mutate(Complaint = gsub("[0-9]", "", Complaint)) |> 
  mutate(Complaint = removePunctuation(Complaint)) |>
  mutate(Complaint = gsub("\n", "", Complaint)) |> 
  mutate(Complaint = gsub("\t", "", Complaint)) |> 
  mutate(Complaint = stripWhitespace(Complaint))

testing_complaints <- Corpus(VectorSource(testing_clean$Complaint)) |>
  tm_map(removeWords, stopwords())

testing_docterm <- DocumentTermMatrix(testing_complaints, list(dictionary = limitfreq)) 

testing_df <- testing_docterm |>
  as.matrix() |>
  as.data.frame()


training_docterm_df <- training_docterm_df |>
  bind_cols(Product=training_clean$Product) |>
  select(Product, everything())

set.seed(1110)
training_df_split <- rsample::initial_split(data = training_docterm_df, prop = 3/4)
train_df <- training(training_df_split)
valid_df <- testing(training_df_split)

train_df_rec <- train_df |>
  recipe(Product ~ .) |>
  step_corr(all_predictors()) |>
  step_nzv(all_predictors())

train_df_model <- rand_forest(mtry = 10, min_n = 4) |>
  set_engine("randomForest") |>
  set_mode("classification")

train_df_wflow <- workflow() |>
  add_recipe(train_df_rec) |>
  add_model(train_df_model)

train_df_wflow

train_df_wflow_fit <- fit(train_df_wflow, data = train_df)

train_df_wflow_fit |>
  pull_workflow_fit()|>
  vip(num_features = 10) 

predict_Product<-predict(train_df_fit, new_data=train_df) 



vfold_df <- rsample::vfold_cv(data = train_df, v = 10)
vfold_df

set.seed(456)

resample_train_df_fit <- tune::fit_resamples(train_df_wflow, vfold_df, control = control_resamples(save_pred = TRUE))
collect_metrics(resample_train_df_fit)

tune_train_df_model <- rand_forest(mode = "classification",
                                   engine = "randomForest",mtry = tune(), min_n = tune())

tune_train_df_wflow <- workflow() |>
  add_recipe(train_df_rec) |>
  add_model(tune_train_df_model)


set.seed(123)
tune_train_df_results <- tune_grid(object = tune_train_df_wflow, resamples = vfold_df, grid  = 20)

tune_train_df_results |>
  collect_metrics() 

show_best(tune_train_df_results, metric = "accuracy", n =1)

tuned_train_df_values<- select_best(tune_train_df_results, "accuracy")
tuned_train_df_values

tuned_train_df_wflow <- tune_train_df_wflow |>
  finalize_workflow(tuned_train_df_values)

overallfit <-tune::last_fit(tuned_train_df_wflow, training_df_split)

collect_metrics(overallfit)

test_predictions <-collect_predictions(overallfit)

ggplot(test_predictions, aes(x=Product, fill=.pred_class)) + 
  geom_bar(position="fill", color="blue") +
  scale_fill_brewer(palette="Set3") + labs(x="Actual Outcome Values",
                                           y="Proportion", fill="Predicted Outcome") +
  theme_bw() +
  theme(axis.text.x=element_text(angle=45, hjust=1, vjust=1))

final_model<-fit(tuned_train_df_wflow, train_df)

predict(final_model, new_data = testing_df)