server <- function(input, output, session) {

  py_require(
    packages = c('numpy', 'scikit-image'),
    python_version = "3.12"
  )

  skimage <- import("skimage.color")
  np <- import("numpy")
  re1 <- reactive({gsub("\\\\", "/", input$upload$datapath)})

  historical_data <- reactiveVal(
    data.frame(
      Image = integer(),
      Model = character(),
      Color_Space = character(),
      Prediction = character(),
      Confidence = numeric(),
      stringsAsFactors = FALSE
    )
  )

  current_id <- reactiveVal(0)

  output$title_image <- renderImage({
    filename <- normalizePath(file.path("./sample/Title.png"))

    list(src = filename, width = "1376px", height = "600px")
    }, deleteFile = FALSE)
  
  observeEvent(input$radio, {
    if (input$radio == 1) {
      img <- reactive({
        f = file.path("./sample/cell_133072_50.png")
        readImage(f)
      })
      output$cell_trivia <- renderText("Early-stage breast cancer; slower growing and less aggressive.")
    } else if (input$radio == 2) {
      img <- reactive({
        f = file.path("./sample/cell_1567_50.png")
        readImage(f)
      })
      output$cell_trivia <- renderText("Early-stage breast cancer; faster growing and more aggressive.")
    } else if (input$radio == 3) {
      img <- reactive({
        f = file.path("./sample/cell_1697_50.png")
        readImage(f)
      })
      output$cell_trivia <- renderText("Cancer cells that have begun invading surrounding tissues.")
    } else if (input$radio == 4) {
      img <- reactive({
        f = file.path("./sample/cell_1379_50.png")
        readImage(f)
      })
      output$cell_trivia <- renderText("Highly aggressive, rapidly proliferating cancer cells.")
    }
    output$dataset_image <- renderDisplay({
      display(img(), method = 'browser')
    })
    observeEvent(input$radio_edit, {
      if (input$radio_edit == 1) {
        output$edit_image <- renderDisplay({
          display(img(), method = 'browser')
        })
        output$colorspace_trivia <- renderText("Original colour from H&E stained cell")
        output$colorspace_image <- renderImage({
          filename <- normalizePath(file.path("./cs_img/rgb.png"))

          list(src = filename,
              width = "400",
              height = "400")
        }, deleteFile = FALSE)
      } else if (input$radio_edit == 2) {
        output$edit_image <- renderDisplay({
          display(channel(img(), "gray"), method = 'browser')
        })
        output$colorspace_trivia <- renderText("Grayscale colour, 
        convert 3 colour channels (R,G,B) to 1 channel by 
        assigning the weight (0.3, 0.59, 0.11) to corresponding colour channel and sum them together to bring out 
        luminosity in the grayscale image, lose colour information but significantly reduce training time")
        output$colorspace_image <- renderImage({
          filename <- normalizePath(file.path("./cs_img/grayscale.png"))

          list(src = filename,
              width = "400",
              height = "200")
        }, deleteFile = FALSE)
      } else if (input$radio_edit == 3) {
        output$edit_image <- renderDisplay({
          rgb_array <- imageData(img())
          rgb_array <- aperm(rgb_array, c(2, 1, 3))  # Fix dimension order
          py_array <- r_to_py(rgb_array, convert = FALSE)
          
          # Convert to CIELAB
          lab_array <- skimage$rgb2lab(py_array)
          
          # Normalize channels for visualization
          lab_norm <- np$asarray(lab_array)
          lab_norm[,,1] <- lab_norm[,,1] / 100        # L*: [0,100] -> [0,1]
          lab_norm[,,2:3] <- (lab_norm[,,2:3] + 128) / 255  # a*/b*: [-128,127] -> [0,1]
          
          # Convert back to EBImage format
          lab_img <- Image(aperm(lab_norm, c(2, 1, 3)), colormode = "Color")
          display(lab_img, method = 'browser')
        })
        output$colorspace_trivia <- renderText("The LAB colour space consists of three channels: L, a, and b. The L channel represents the lightness or brightness of a colour and ranges from 0 (black) to 100 (white).  The a channel represents the colour on a red-green axis, with positive values representing red and negative values representing green. The b channel represents the colour on a blue-yellow axis, with positive values representing yellow and negative values representing blue.")
        output$colorspace_image <- renderImage({
          filename <- normalizePath(file.path("./cs_img/cielab.png"))

          list(src = filename,
              width = "400",
              height = "340")
        }, deleteFile = FALSE)
      } else if (input$radio_edit == 4) {
        output$edit_image <- renderDisplay({
          # Convert to numpy array [0-255]
          rgb_array <- imageData(img())
          rgb_array <- aperm(rgb_array, c(2, 1, 3))  # Fix dimension order
          py_array <- r_to_py(rgb_array, convert = FALSE)
          
          # Convert to YCbCr (output range: Y 16-235, Cb/Cr 16-240)
          ycbcr_array <- skimage$rgb2ycbcr(py_array)
          
          # Normalize channels for visualization
          y_channel <- ycbcr_array[,,1] / 255
          cb_channel <- (ycbcr_array[,,2] - 16) / (240 - 16)
          cr_channel <- (ycbcr_array[,,3] - 16) / (240 - 16)
          
          # Combine normalized channels
          ycbcr_norm <- np$dstack(list(y_channel, cb_channel+150, cr_channel))
          
          # Convert back to EBImage format
          ycbcr_img <- Image(aperm(np$asarray(ycbcr_norm), c(2, 1, 3)), 
                            colormode = "Color")
          display(ycbcr_img, method = 'browser')
        })
        output$colorspace_trivia <- renderText("The YCbCr color space consists of three channels: Y, Cb, and Cr. The Y channel represents the luma or brightness component, and the CB and CR channels represent the chroma or color information and are calculated as the difference between the luma component and the blue and red components, respectively .")
        output$colorspace_image <- renderImage({
          filename <- normalizePath(file.path("./cs_img/133072 ycbcr.png"))

          list(src = filename,
              width = "400",
              height = "124")
        }, deleteFile = FALSE)
      }
    })
  })

  output$cell_image <- renderImage({
    req(input$upload) # Ensure a file is uploaded
    list(src = re1(), width = "200px", height = "200px")
    }, deleteFile = FALSE)

  output[["predict.button"]] <- renderUI({
    req(input$upload)
    actionButton("predict.image", "Classify", style = "white-space:normal")
  })
  observeEvent(input$upload, {
    current_id(current_id() + 1)
    img = readImage(input$upload$datapath)
    observeEvent(input$color_predict, {
    if (input$color_predict == "cielab") {
        output$edit_predict <- renderDisplay({
          rgb_array <- imageData(img)
          rgb_array <- aperm(rgb_array, c(2, 1, 3))  # Fix dimension order
          py_array <- r_to_py(rgb_array, convert = FALSE)
          
          # Convert to CIELAB
          lab_array <- skimage$rgb2lab(py_array)
          
          # Normalize channels for visualization
          lab_norm <- np$asarray(lab_array)
          lab_norm[,,1] <- lab_norm[,,1] / 100        # L*: [0,100] -> [0,1]
          lab_norm[,,2:3] <- (lab_norm[,,2:3] + 128) / 255  # a*/b*: [-128,127] -> [0,1]
          
          # Convert back to EBImage format
          lab_img <- Image(aperm(lab_norm, c(2, 1, 3)), colormode = "Color")
          display(lab_img, method = 'browser')
        })
      } else if (input$color_predict == "ycbcr") {
        output$edit_predict <- renderDisplay({
          rgb_array <- imageData(img)
          rgb_array <- aperm(rgb_array, c(2, 1, 3))  # Fix dimension order
          py_array <- r_to_py(rgb_array, convert = FALSE)
          
          # Convert to YCbCr (output range: Y 16-235, Cb/Cr 16-240)
          ycbcr_array <- skimage$rgb2ycbcr(py_array)
          
          # Normalize channels for visualization
          y_channel <- ycbcr_array[,,1] / 255
          cb_channel <- (ycbcr_array[,,2] - 16) / (240 - 16)
          cr_channel <- (ycbcr_array[,,3] - 16) / (240 - 16)

          # Combine normalized channels
          ycbcr_norm <- np$dstack(list(y_channel, cb_channel+150, cr_channel))
          
          # Convert back to EBImage format
          ycbcr_img <- Image(aperm(np$asarray(ycbcr_norm), c(2, 1, 3)), 
                            colormode = "Color")
          display(ycbcr_img, method = 'browser')
        })
    } else if (input$color_predict == "grayscale") {
      output$edit_predict <- renderDisplay({
        display(channel(img, "gray"), method = 'browser')
      })
    } else {
      output$edit_predict <- renderDisplay({
        display(img, method = 'browser')
      })
    } 
  })
  })
  observeEvent(input$predict.image, {
    class_labels <- c("DCIS1", "DCIS2", "Invasive Tumour", "Prolif Invasive Tumour")
    model_path = file.path(".", "models", input$model_predict, input$color_predict, "model_best.keras")
    source_python("predict_cli.py")
    predicted_class <- model_predict(model_path, input$upload$datapath, input$color_predict)
    output$predict_output_1 <- renderText({
      paste("Prediction:", class_labels[predicted_class[[1]] + 1])
    })
    output$predict_output_2 <- renderText({
      paste("Confidence:", round(max(predicted_class[[2]]) * 100, 1), "%")
    })
    new_entry <- data.frame(
      Image = current_id(),
      Model = input$model_predict,
      Color_Space = input$color_predict,
      Prediction = class_labels[predicted_class[[1]] + 1],
      Confidence = round(max(predicted_class[[2]]) * 100, 1),
      stringsAsFactors = FALSE
    )
    historical_data(rbind(new_entry, historical_data()))
    output$history <- renderDT({
      historical_data()
      },
      filter = 'top',
      options = list(
        pageLength = 5,
        bLengthChange = FALSE,
        autoWidth = TRUE
      ),
      rownames = FALSE
    )
  })

  output$predict_text <- renderText({ 
    "Upload a cell image, select between 2 high performing CNN models, select color space and click \"Classify\" to classify the type of tumour cell using the selected model" 
  })

  output$introduction_text <- renderText({ 
    "Impact of Colour Space Transformations on Tumor Cell Classification with CNN and Random Forest Model" 
  })
  zoom <- reactiveValues(xlim = NULL, ylim = NULL)
  observeEvent(input$plot_brush, {
    zoom$xlim <- c(input$plot_brush$xmin, input$plot_brush$xmax)
    zoom$ylim <- c(input$plot_brush$ymin, input$plot_brush$ymax)
  })
  observeEvent(input$reset_zoom, {
    zoom$xlim <- NULL
    zoom$ylim <- NULL
  })
  output$plot <- renderPlot({
    data <- read.csv("training_data/avg_model_performance.csv", header = TRUE, check.names = FALSE)
    # Rename columns and clean data
    colnames(data) <- c("Model", "ColorSpace", "Accuracy", "Precision", "Time")
    data <- data %>% fill(Model)  # Fill missing model names

    # Standardize naming conventions
    data$Model <- factor(data$Model,
                        levels = c("RF", "ResNet", "Mobilenet", "Efficientnet"),
                        labels = c("RF", "ResNet", "MobileNet", "EfficientNet"))
    data$ColorSpace <- gsub("Gray scale", "Grayscale", data$ColorSpace)
    data$ColorSpace <- factor(data$ColorSpace, 
                              levels = c("RGB", "Grayscale", "CIELAB", "YCbCr"))

    # Filter data selection
    model_filter <- input$model_select
    color_filter <- input$color_select
    data_subset <- data %>% 
      filter(Model %in% model_filter)
    data_final <- data_subset %>% 
      filter(ColorSpace %in% color_filter)
    
    # Create the plot
    y_metric <- input$y_axis
    if (y_metric == "Time") {
      dec_place <- "%.1f"
      label <- "Time (seconds)"
    } else {
      dec_place <- "%.3f"
      label <- y_metric
    }
    p <- ggplot(data_final, aes(x = Model, y = !!sym(y_metric), fill = ColorSpace)) +
      geom_col(position = position_dodge(0.8), width = 0.7) +
      geom_text(aes(label = sprintf(dec_place, !!sym(y_metric))),  # Format decimal places
                position = position_dodge(width = 0.8),   # Match bar dodge width
                vjust = -0.3,                            # Position above bars
                size = 3.5,
                fontface="bold") +                               # Adjust text size
      scale_fill_manual(values = c(
        "RGB" = "#FF0000",
        "Grayscale" = "#808080",
        "CIELAB" = "#0072B2",
        "YCbCr" = "#009E73"
      )) +
      labs(x = "Model", y = label, fill = "Colour Space") +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1, face="bold"),
            axis.title.x = element_text(face="bold", colour="red", size = 20),
            axis.title.y = element_text(face="bold", colour="red", size = 20),
            legend.title = element_text(face="bold", size = 10)) +
      scale_y_continuous(expand = expansion(mult = c(0, 0.15)))  # Add space for labels
      if (!is.null(zoom$xlim) && !is.null(zoom$ylim)) {
        p <- p + coord_cartesian(xlim = zoom$xlim, ylim = zoom$ylim)
      }
      p
  })
}
