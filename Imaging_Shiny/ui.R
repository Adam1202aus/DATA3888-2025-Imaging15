ui <- dashboardPage(
  skin = "purple",
  

  # App Header Title
  dashboardHeader(title = p(icon("image"), span("Cell classify", style = "font-weight: bold;"))) |>
    # Middle Title addition reference from: https://stackoverflow.com/questions/75269898/position-title-in-the-header-in-shinydashboard
    tagAppendChild(
      div(
        "Impact of Colour Space Transformations on Tumor Cell Classification with CNN and Random Forest Model",
        style = "
      display: block;
      font-size: 1.5em;
      margin-block-start: 0.5em;
      font-weight: bold;
      color: white;
      margin-right: 10%",
        align = "right"
      ),
      .cssSelector = "nav"
    ),
  dashboardSidebar(
    collapsed = TRUE,
    sidebarMenu(
    # Setting id makes input$tabs give the tabName of currently-selected tab
    id = "tabs",
    menuItem("Introduction", tabName = "introduction", icon = icon("plus")),
    menuItem("Transformation", tabName = "process", icon = icon("image")),
    menuItem("Models", tabName = "demo", icon = icon("brain")),
    menuItem("Predict", tabName = "predict", icon = icon("magnifying-glass-chart"))
    )
  ),
  dashboardBody(
    tags$head(
      tags$style(HTML("
        .ttip {
          position: relative;
          display: inline-block;
        }
        
        .ttip .tooltiptext {
          visibility: hidden;
          width: 160px;
          background-color: #555;
          color: #fff;
          text-align: center;
          border-radius: 6px;
          padding: 5px;
          position: absolute;
          z-index: 10000;
          left: 110%;
          opacity: 0;
          transition: opacity 0.3s;
        }
        
        .ttip:hover .tooltiptext {
          visibility: visible;
          opacity: 1;
        }

        .fa-info-circle {
          font-size: 0.8em;
          color: #3498db;
          margin-left: 5px;
        }

        .ttip:hover .fa-info-circle {
          color: #1f618d;
        }
      "))
    ),
    tabItems(
      # First tab content
      tabItem(tabName = "introduction",
        imageOutput("title_image")
      ),

      # Second tab content
      tabItem(tabName = "process",
        fluidRow(
          column(width = 4,
            radioButtons( 
              inputId = "radio", 
              label = "1. Select cell type", 
              choices = list( 
                "DCIS 1" = 1, 
                "DCIS 2" = 2,
                "Invasive Tumor" = 3,
                "Prolif Invasive Tumor" = 4
              ) 
            ), 
            displayOutput("dataset_image", width = "400px", height = "400px"),
            textOutput("cell_trivia")
          ),
          column(width = 4,
            radioButtons( 
              inputId = "radio_edit", 
              label = "2. Select color space", 
              choices = list( 
                "RGB (Original)" = 1, 
                "Grayscale" = 2,
                "CIELAB" = 3,
                "YCBCR" = 4
              ) 
            ), 
            displayOutput("edit_image", width = "400px", height = "400px"),
          ),
          column(width = 4,
            box(
              title = "Color space info",
              status = "success",
              solidHeader = TRUE,
              collapsible = FALSE,
              collapsed = FALSE,
              width = 12,
              card(
                textOutput("colorspace_trivia"),
                imageOutput("colorspace_image")
              )
            )
          )
        )
      ),

      # Third tab content
      tabItem(tabName = "demo",
        sidebarLayout(
          sidebarPanel(
            checkboxGroupInput("model_select", "Select models to compare:",  
                  choiceNames = list(
                    HTML('<div class="ttip">RF <i class="fas fa-info-circle"></i><span class="tooltiptext">Random Forest Model: Ensemble of decision trees using majority voting for robust predictions. Handles high-dimensional data well, resistant to overfitting, suitable for classification/regression tasks.</span></div>'),
                    HTML('<div class="ttip">ResNet <i class="fas fa-info-circle"></i><span class="tooltiptext">Residual Network Model: Deep CNN with residual connections enabling training of 100+ layers. Solves vanishing gradients via skip connections, excels in image recognition.</span></div>'),
                    HTML('<div class="ttip">MobileNet <i class="fas fa-info-circle"></i><span class="tooltiptext">MobileNetV2 CNN Model: Lightweight CNN using depthwise separable convolutions for mobile/embedded devices. Optimizes speed/accuracy trade-off via parameter reduction.</span></div>'),
                    HTML('<div class="ttip">EfficientNet <i class="fas fa-info-circle"></i><span class="tooltiptext">EfficientNetB0 Model: Scalable CNN balancing depth/width/resolution via compound scaling. Maximizes accuracy per computational budget, outperforms manual architectures.</span></div>')
                      ),
                  choiceValues = c("RF", "ResNet", "MobileNet", "EfficientNet"),
                  selected = c("RF",
                  "ResNet",
                  "MobileNet",
                  "EfficientNet")),
            checkboxGroupInput("color_select", "Select colour space to compare:",
                  inline = TRUE,  
                  choices = c("RGB",
                  "Grayscale",
                  "CIELAB",
                  "YCbCr"),
                  selected = c("RGB",
                  "Grayscale",
                  "CIELAB",
                  "YCbCr")),
            selectInput("y_axis", "Select result (Y-axis):",  
                  choices = c("Precision",
                  "Accuracy",
                  "Time"),
                  selected = "Precision")
          ),
          mainPanel(
            plotOutput("plot", brush = brushOpts(id = "plot_brush", resetOnNew = TRUE)),
            actionButton("reset_zoom", "Reset Zoom")
          )
        )
      ),
      

      # Fourth tab content
      tabItem(tabName = "predict",
        sidebarLayout(
          sidebarPanel(
            textOutput("predict_text"),
            fileInput("upload", "1. Upload an image", accept = c("image/png", "image/jpeg", "image/jpg")),
            imageOutput("cell_image", width = "200px", height = "200px"),
            # ResNet was not used to the model's big size which would lead
            # to crash in R Shiny, and its low accuracy. Random Forest were not used
            # due to very low accuracy, basically random guessing.
            selectInput("model_predict", "2. Select a CNN model:",  
                  choices = c("MobileNetV2" = "mobnet",
                  "EfficientNetB0" = "effnet"),
                  selected = "effnet"), 
            selectInput("color_predict", "3. Select colorspace:",  
                  choices = c("RGB" = "original",
                  "Grayscale" = "grayscale",
                  "CIELAB" = "cielab",
                  "YCbCr" = "ycbcr"),
                  selected = "ycbcr"), 
          ),
          mainPanel(width = 8,
            column(width = 4,displayOutput("edit_predict", width = "300px", height = "300px")),
            column(width = 8,
              uiOutput("predict.button"),
              h2(textOutput("predict_output_1")),
              h2(textOutput("predict_output_2"))),
            DT::dataTableOutput("history")
          )
        )
      )
    )
  )

)