


#LOAD REQUIRED LIBRARIES -----------------------------------------------------------------


library(shiny)
library(ggplot2)
library(tidymodels)
library(tidyr)
library(rpart.plot)
library(ranger)
library(bslib)
library(bsicons)


# LOAD DATA ---------------------------------------------------------------


setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
addResourcePath("www", file.path(getwd(), "www"))

h <- read.csv("Heart_disease_cleveland_new.csv")
h$target <- as.factor(h$target)



# UI ----------------------------------------------------------------------


ui <- page_sidebar(
  
  sidebar = sidebar(
    
    sliderInput("split_ratio", "Training Data Split:",
                min = 0.1, max = 0.99, value = 0.5),
    
    conditionalPanel(
      condition = "input.tabs == 'Decision Tree'",
      sliderInput(
        inputId = "tree_depth",
        label = tooltip(
          trigger = list("Tree Depth:", bsicons::bs_icon("info-circle")),
          "The tree may not always change with depth because rpart 
           automatically prunes branches that do not improve accuracy. 
           Pruning removes unnecessary splits to avoid overfitting."
        ),
        min = 1, max = 30, value = 5
      ),
      img(src = "www/tree_legend.png", width = "100%")
    ),
    
    conditionalPanel(
      condition = "input.tabs == 'Random Forest'",
        sliderInput(inputId ="rf_trees",label = tooltip(
            trigger = list( "Number of Trees:", bsicons::bs_icon("info-circle")),
              "How many decision trees the Random Forest builds. 
               More trees usually improve 
               stability but take longer to run."
                                           ),
                  min = 50, max = 500, value = 100
                  ),
      
      
      sliderInput(inputId ="rf_mtry", label = tooltip(
              trigger = list("Variables per Split (mtry):", bsicons::bs_icon("info-circle")),
              "How many features the model randomly 
               considers each time it looks for the
               best split in a tree."
      ),
                  min = 1, max = 13, value = 3),
      sliderInput(inputId ="rf_min_n", label = tooltip(
        trigger = list("Minimum Node Size:", bsicons::bs_icon("info-circle")),
        "The smallest number of observations allowed in a final node (leaf).
        Larger values make the model simpler; 
        smaller values allow more detailed splits."
      ),
                  min = 1, max = 20, value = 5)
    )
    
  ),  # sidebar closes here
  
  tabsetPanel(
    id = "tabs",
    
    tabPanel(
      "Decision Tree",
      card(
        card_header("Model Performance Metrics"),
        tableOutput("dt_metrics_table")
      ),
      plotOutput("dt_plot")
    ),
    
    tabPanel(
      
      
      "Random Forest",
      card(
        card_header("Model Performance Metrics"),
        tableOutput("rf_metrics_table")
      ),
      helpText(
        "Random Forest (Classification):
        A machine learning method that builds many decision trees on
        random samples of the data and predicts the class by taking
        the majority vote across all trees, producing a more accurate
        and stable classification."
      ),
      plotOutput("rf_plot")
      
    )
    
  )
  
)  # page_sidebar closes here




# SERVER ------------------------------------------------------------------


server <- function(input, output) {
  
  ds <- reactive({
    set.seed(123)
    data_split <- initial_split(h, prop = input$split_ratio)
    list(
      train_data = training(data_split),
      test_data  = testing(data_split)
    )
  })
  
  # Decision Tree
  model <- reactive({
    tree_spec <- decision_tree(tree_depth = input$tree_depth) %>%
      set_engine("rpart") %>%
      set_mode("classification")
    
    tree_fit <- tree_spec %>%
      fit(target ~ ., data = ds()$train_data)
    
    predictions <- tree_fit %>%
      predict(ds()$test_data) %>%
      pull(.pred_class)
    
    results <- ds()$test_data %>%
      mutate(predicted = predictions)
    
    TP <- sum(results$target == 1 & results$predicted == 1)
    TN <- sum(results$target == 0 & results$predicted == 0)
    FP <- sum(results$target == 0 & results$predicted == 1)
    FN <- sum(results$target == 1 & results$predicted == 0)
    
    list(
      accuracy_dt    = (TP + TN) / (TP + TN + FP + FN),
      sensitivity_dt = TP / (TP + FN),
      specificity_dt = TN / (TN + FP),
      precision_dt   = TP / (TP + FP),
      tree_fit       = tree_fit
    )
  })
  
  output$dt_metrics_table <- renderTable({
    data.frame(
      metric = c("Accuracy", "Sensitivity", "Specificity", "Precision"),
      value  = c(model()$accuracy_dt, model()$sensitivity_dt,
                 model()$specificity_dt, model()$precision_dt)
    )
  })
  
  output$dt_plot <- renderPlot({
    rpart.plot(model()$tree_fit$fit,
               type = 2, extra = 104,
               fallen.leaves = TRUE,
               main = "Decision Tree for Heart Disease")
  })
  
  # Random Forest
  rf <- reactive({
    train <- ds()$train_data
    test  <- ds()$test_data
    
    rf_spec <- rand_forest(
      trees = input$rf_trees,
      mtry  = input$rf_mtry,
      min_n = input$rf_min_n
    ) %>%
      set_engine("ranger") %>%
      set_mode("classification")
    
    rf_fit <- rf_spec %>%
      fit(target ~ ., data = train)
    
    rf_predictions <- rf_fit %>%
      predict(test) %>%
      pull(.pred_class)
    
    results <- test %>%
      mutate(predicted = rf_predictions)
    
    TP <- sum(results$target == 1 & results$predicted == 1)
    TN <- sum(results$target == 0 & results$predicted == 0)
    FP <- sum(results$target == 0 & results$predicted == 1)
    FN <- sum(results$target == 1 & results$predicted == 0)
    
    list(
      accuracy_rf    = (TP + TN) / (TP + TN + FP + FN),
      sensitivity_rf = TP / (TP + FN),
      specificity_rf = TN / (TN + FP),
      precision_rf   = TP / (TP + FP)
    )
  })
  
  output$rf_metrics_table <- renderTable({
    data.frame(
      metric = c("Accuracy", "Sensitivity", "Specificity", "Precision"),
      value  = c(rf()$accuracy_rf, rf()$sensitivity_rf,
                 rf()$specificity_rf, rf()$precision_rf)
    )
  })
  
}



# RUN THE APP -------------------------------------------------------------


shinyApp(ui = ui, server = server)