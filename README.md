# DATA3888 2025 Imaging15 Reproducible guide
## 1. Data cleaning, extracting
Data cleaning, extracting code is located inside "Model Training" folder
TODO:

## 2. Model training
Model training code is located inside "Model Training" folder
TODO:

## 3. Final Report's table
The final report's R code will extract the analysis data generated during model training, specifically in the "report.txt" and "training_time.txt" file, located inside "DATA Models/*model_type*/*colourspace*/fold *" folder to create a table showing the average performance metrics (accuracy, precision, time) across all folds. The code will also create a "avg_model_performance.csv" file that can be used to generate the side-by-side plot in R Shiny application by putting the csv file inside "Imaging_Shiny/training_data"

## 4. R Shiny deployment
### Required packages
The whole R Shiny app is located inside "Imaging_Shiny"
Required packages are listed inside "Imaging_Shiny/global.R"

### Online deployment
The application can be deployed to R Shiny through shinyapps.io by using the "rsconnect" package, and following the guide on this shinyapps.io site: https://docs.posit.co/shinyapps.io/guide/getting_started/
