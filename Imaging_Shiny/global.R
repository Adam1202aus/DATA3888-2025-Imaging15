# Library to used with the Shiny app
library(shiny)
library(bslib)
library(shinydashboard)
library(shinydashboardPlus)
library(DT)
library(ggplot2)
library(tidyr)
library(dplyr)

# Use "reticulate::py_install("scikit-image")" in R terminal to install 
# scikit-image library
library(reticulate)


# Use the below command to install tools for managing Python virtual environments
# "sudo apt-get install python3-venv python3-pip python3-dev"
# Make sure to upgrade keras in python
# 'pip install --upgrade keras'
library(keras3)


# To install EBImage in Linux Ubuntu, you have to install fftw3 on your OS
# "sudo apt-get install fftw3 fftw3-dev pkg-config",
# then install.packages("ffwtools") in R
library(EBImage)
