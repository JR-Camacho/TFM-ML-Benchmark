library(caret)
library(pROC)
library(xgboost)
library(ggplot2)
library(dplyr)
library(plotly)
library(htmltools)

entrenar_benchmark <- function(train_data, nombre_dataset, scaled_df = FALSE) {
  cat("\n======================================================\n")
  cat("--- Iniciando Entrenamiento para:", nombre_dataset, "---\n")
  cat("======================================================\n")
  
  t_inicio_total <- Sys.time()
  tiempos_modelos <- list()
  
  ctrl <- trainControl(
    method = "cv", 
    number = 5, 
    classProbs = TRUE, 
    summaryFunction = twoClassSummary,
    verboseIter = TRUE 
  )
  
  algoritmos <- list(
    LogisticRegression = "glm",
    DecisionTree = "rpart",
    RandomForest = "rf",
    GBM = "gbm",
    C50 = "C5.0"
  )
  
  if (scaled_df == TRUE) {
    algoritmos$KNN <- "knn"
  }
  
  modelos_entrenados <- list()
  
  for (nombre in names(algoritmos)) {
    cat("\n------------------------------------------------------\n")
    cat(">> Entrenando Modelo:", nombre, "con método '", algoritmos[[nombre]], "'...\n")
    cat("------------------------------------------------------\n")
    
    t_inicio_mod <- Sys.time()
    
    modelos_entrenados[[nombre]] <- train(
      hospdead ~ ., 
      data = train_data, 
      method = algoritmos[[nombre]], 
      metric = "ROC", 
      trControl = ctrl
    )
    
    t_fin_mod <- Sys.time()
    tiempo_mod <- round(as.numeric(difftime(t_fin_mod, t_inicio_mod, units = "mins")), 2)
    tiempos_modelos[[nombre]] <- tiempo_mod
    
    cat(">> Completado:", nombre, "| Tiempo:", tiempo_mod, "minutos\n")
  }
  
  t_fin_total <- Sys.time()
  tiempo_total <- round(as.numeric(difftime(t_fin_total, t_inicio_total, units = "mins")), 2)
  
  cat("\n======================================================\n")
  cat("RESUMEN DE TIEMPOS DE ENTRENAMIENTO\n")
  for (nombre in names(tiempos_modelos)) {
    cat(nombre, ":", tiempos_modelos[[nombre]], "minutos\n")
  }
  cat("TIEMPO TOTAL:", tiempo_total, "minutos\n")
  cat("======================================================\n")
  
  return(modelos_entrenados)
}


evaluar_benchmark <- function(modelos_lista, test_data, nombre_dataset, clase_positiva = "Yes") {
  cat("\n--- Evaluando:", nombre_dataset, "en el set de TEST ---\n")
  
  resultados_finales <- data.frame()
  
  for (nombre in names(modelos_lista)) {
    modelo <- modelos_lista[[nombre]]
    
    # 1. Obtener predicciones
    # Usamos na.action = na.pass para intentar predecir incluso con valores extraños
    probs_df <- predict(modelo, test_data, type = "prob")
    clases <- predict(modelo, test_data)
    
    # 2. Sincronización estricta:
    # Solo tomamos las filas que realmente fueron predichas (en caso de que el modelo omita alguna)
    filas_validas <- !is.na(clases) 
    
    y_test_valid <- test_data$hospdead[filas_validas]
    probs_valid <- probs_df[filas_validas, clase_positiva]
    clases_valid <- clases[filas_validas]
    
    # 3. Calcular métricas con los datos sincronizados
    roc_obj <- roc(y_test_valid, probs_valid, quiet = TRUE)
    cm <- confusionMatrix(clases_valid, y_test_valid, positive = clase_positiva)
    
    fila <- data.frame(
      Dataset = nombre_dataset,
      Modelo = nombre,
      AUC = as.numeric(auc(roc_obj)),
      Sensibilidad = cm$byClass["Sensitivity"],
      Especificidad = cm$byClass["Specificity"],
      Accuracy = cm$overall["Accuracy"]
    )
    resultados_finales <- rbind(resultados_finales, fila)
  }
  
  return(resultados_finales)
}

generar_graficos_automaticos <- function(resultados) {
  # 1. Detectamos automáticamente los datasets y definimos las métricas a evaluar
  datasets_unicos <- unique(resultados$Dataset)
  metricas <- c("AUC", "Sensibilidad", "Especificidad", "Accuracy")
  
  # Lista para almacenar todos los gráficos generados
  lista_graficos <- list()
  
  # 2. Bucle anidado: Por cada dataset, generamos las 4 métricas
  for (ds in datasets_unicos) {
    datos_ds <- resultados %>% filter(Dataset == ds)
    
    for (met in metricas) {
      
      # Encontrar el modelo ganador para esta métrica específica (para pintarlo de azul)
      modelo_max <- datos_ds$Modelo[which.max(datos_ds[[met]])]
      
      # Usamos .data[[met]] para que ggplot lea el nombre de la columna dinámicamente
      p <- ggplot(datos_ds, aes(x = reorder(Modelo, .data[[met]]), 
                                y = .data[[met]], 
                                fill = Modelo == modelo_max, 
                                text = paste("Modelo:", Modelo, "<br>", met, ":", round(.data[[met]], 3)))) +
        geom_bar(stat = "identity", width = 0.6) +
        scale_fill_manual(values = c("TRUE" = "#abc4ff", "FALSE" = "#a8e69d")) + 
        theme_minimal() +
        theme(
          plot.background = element_rect(fill = "#121212", color = NA),
          panel.background = element_rect(fill = "#121212", color = NA),
          text = element_text(color = "white"),
          axis.text = element_text(color = "gray80"),
          axis.text.x = element_text(size = 12, face = "bold"),
          panel.grid.major.y = element_line(color = "gray30"),
          panel.grid.minor = element_blank(),
          panel.grid.major.x = element_blank(),
          legend.position = "none",
          plot.title = element_text(size = 16, face = "bold", margin = margin(b = 5)),
          plot.subtitle = element_text(size = 12, color = "gray50", margin = margin(b = 15))
        ) +
        labs(
          title = paste("Comparativa de", met),
          subtitle = paste("Dataset:", ds),
          x = "",
          y = paste("Valor de", met, "↑")
        ) +
        coord_cartesian(ylim = c(0, 1))
      
      # Convertir a Plotly interactivo
      p_interactivo <- ggplotly(p, tooltip = "text") %>%
        layout(plot_bgcolor = "#121212", paper_bgcolor = "#121212")
      
      # Guardar el gráfico en la lista
      lista_graficos[[paste(ds, met, sep = "_")]] <- p_interactivo
    }
  }
  
  # 3. Empaquetar todo para que RMarkdown lo renderice de golpe en el HTML final
  return(htmltools::browsable(htmltools::tagList(lista_graficos)))
}

plot_roc_comparativo <- function(modelos_lista, test_data, titulo = "Comparación de Curvas ROC") {
  
  lista_roc <- list()
  
  for (nombre in names(modelos_lista)) {
    modelo <- modelos_lista[[nombre]]
    
    # Obtenemos predicciones
    preds <- predict(modelo, test_data, type = "prob")
    
    # Filtramos filas para evitar errores por NAs (crítico para modelos escalados)
    filas_validas <- !is.na(preds$Yes)
    y_test_valid <- test_data$hospdead[filas_validas]
    prob_valid <- preds$Yes[filas_validas]
    
    # Guardamos el objeto ROC
    lista_roc[[nombre]] <- roc(y_test_valid, prob_valid, quiet = TRUE)
  }
  
  # Generar la gráfica
  grafica <- ggroc(lista_roc, legacy.axes = TRUE) +
    theme_minimal() +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "grey") +
    labs(title = titulo,
         x = "Tasa de Falsos Positivos (1-Especificidad)",
         y = "Tasa de Verdaderos Positivos (Sensibilidad)") +
    theme(legend.title = element_blank(),
          plot.title = element_text(face = "bold", size = 14))
  
  return(grafica)
}