.check_required_packages <- function(packages) {
  missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) > 0) {
    stop(
      "Faltan paquetes requeridos para modeling_utils_tidymodels.R: ",
      paste(missing, collapse = ", "),
      ". Instalar antes de entrenar con install.packages(c(\"",
      paste(missing, collapse = "\", \""),
      "\"))",
      call. = FALSE
    )
  }
}

.positive_event_level <- function(data, target = "hospdead", clase_positiva = "Yes") {
  if (!target %in% names(data)) {
    stop("No se encontro la variable objetivo '", target, "'.", call. = FALSE)
  }

  if (!is.factor(data[[target]])) {
    data[[target]] <- as.factor(data[[target]])
  }

  if (!clase_positiva %in% levels(data[[target]])) {
    stop(
      "La clase positiva '", clase_positiva,
      "' no existe en los niveles de ", target, ": ",
      paste(levels(data[[target]]), collapse = ", "),
      call. = FALSE
    )
  }

  data[[target]] <- stats::relevel(data[[target]], ref = clase_positiva)
  data
}

.make_metric_set <- function() {
  yardstick::metric_set(
    yardstick::roc_auc,
    yardstick::sens,
    yardstick::spec,
    yardstick::accuracy
  )
}

.cv_metric_set <- function(nombre) {
  if (identical(nombre, "SVM")) {
    return(yardstick::metric_set(yardstick::roc_auc))
  }

  .make_metric_set()
}

.roc_for_positive_class <- function(truth, prob_positive, clase_positiva) {
  negative_levels <- setdiff(levels(truth), clase_positiva)
  if (length(negative_levels) != 1) {
    stop(
      "Se esperaba una clasificacion binaria con una sola clase negativa. Niveles: ",
      paste(levels(truth), collapse = ", "),
      call. = FALSE
    )
  }

  pROC::roc(
    truth,
    prob_positive,
    levels = c(negative_levels, clase_positiva),
    quiet = TRUE
  )
}

.modelo_specs <- function(scaled_df = FALSE, incluir_svm = TRUE) {
  specs <- list(
    LogisticRegression = list(
      spec = parsnip::logistic_reg(mode = "classification") |>
        parsnip::set_engine("glm"),
      tune = FALSE
    ),

    DecisionTree = list(
      spec = parsnip::decision_tree(
        mode = "classification",
        cost_complexity = tune::tune(),
        tree_depth = tune::tune(),
        min_n = tune::tune()
      ) |>
        parsnip::set_engine("rpart"),
      tune = TRUE
    ),

    RandomForest = list(
      spec = parsnip::rand_forest(
        mode = "classification",
        mtry = tune::tune(),
        trees = 500,
        min_n = tune::tune()
      ) |>
        parsnip::set_engine("ranger", importance = "impurity", probability = TRUE),
      tune = TRUE
    ),

    GBM = list(
      spec = parsnip::boost_tree(
        mode = "classification",
        trees = tune::tune(),
        tree_depth = tune::tune(),
        learn_rate = tune::tune(),
        loss_reduction = tune::tune(),
        min_n = tune::tune()
      ) |>
        parsnip::set_engine("xgboost"),
      tune = TRUE
    )
  )

  if (incluir_svm == TRUE) {
    specs$SVM <- list(
      spec = parsnip::svm_rbf(
        mode = "classification",
        cost = tune::tune(),
        rbf_sigma = tune::tune()
      ) |>
        parsnip::set_engine("kernlab"),
      tune = TRUE
    )
  }

  if (scaled_df == TRUE) {
    specs$KNN <- list(
      spec = parsnip::nearest_neighbor(
        mode = "classification",
        neighbors = tune::tune(),
        weight_func = tune::tune(),
        dist_power = tune::tune()
      ) |>
        parsnip::set_engine("kknn"),
      tune = TRUE
    )
  }

  specs
}

.default_params_no_cv <- function(nombre, train_data) {
  n_predictores <- ncol(train_data) - 1

  switch(
    nombre,
    DecisionTree = tibble::tibble(
      cost_complexity = 0.01,
      tree_depth = 30L,
      min_n = 20L
    ),
    RandomForest = tibble::tibble(
      mtry = max(1L, floor(sqrt(n_predictores))),
      min_n = 5L
    ),
    GBM = tibble::tibble(
      trees = 100L,
      tree_depth = 3L,
      learn_rate = 0.1,
      loss_reduction = 0,
      min_n = 10L
    ),
    SVM = tibble::tibble(
      cost = 1,
      rbf_sigma = 0.01
    ),
    KNN = tibble::tibble(
      neighbors = 5L,
      weight_func = "rectangular",
      dist_power = 2
    ),
    NULL
  )
}

.fit_tidymodels_model <- function(nombre, spec, train_data, usar_cv, folds, metricas,
                                  grid_size, tune_model) {
  wf <- workflows::workflow() |>
    workflows::add_formula(hospdead ~ .) |>
    workflows::add_model(spec)

  if (usar_cv == TRUE && tune_model == TRUE) {
    tuned <- tune::tune_grid(
      wf,
      resamples = folds,
      grid = grid_size,
      metrics = .cv_metric_set(nombre),
      control = tune::control_grid(save_pred = FALSE, verbose = TRUE)
    )

    best_params <- tune::select_best(tuned, metric = "roc_auc")
    final_wf <- tune::finalize_workflow(wf, best_params)
    final_fit <- parsnip::fit(final_wf, data = train_data)

    return(list(
      engine = "tidymodels",
      nombre = nombre,
      workflow = final_wf,
      fit = final_fit,
      resamples = tuned,
      best_params = best_params
    ))
  }

  if (usar_cv == FALSE && tune_model == TRUE) {
    default_params <- .default_params_no_cv(nombre, train_data)
    if (is.null(default_params)) {
      stop("No hay hiperparametros por defecto definidos para ", nombre, ".", call. = FALSE)
    }

    wf <- tune::finalize_workflow(wf, default_params)
  }

  final_fit <- parsnip::fit(wf, data = train_data)

  list(
    engine = "tidymodels",
    nombre = nombre,
    workflow = wf,
    fit = final_fit,
    resamples = NULL,
    best_params = NULL
  )
}

entrenar_benchmark <- function(train_data, nombre_dataset, scaled_df = FALSE, usar_cv = TRUE,
                               folds = 5, grid_size = 10, clase_positiva = "Yes",
                               incluir_svm = TRUE) {
  .check_required_packages(c(
    "tidymodels", "parsnip", "workflows", "tune", "rsample",
    "yardstick", "dplyr", "ggplot2", "plotly", "htmltools",
    "pROC", "rpart", "ranger", "xgboost", "tibble"
  ))

  if (scaled_df == TRUE) {
    .check_required_packages(c("kknn"))
  }

  if (incluir_svm == TRUE) {
    .check_required_packages(c("kernlab"))
  }

  cat("\n======================================================\n")
  cat("--- Iniciando Entrenamiento tidymodels para:", nombre_dataset, "---\n")
  cat("======================================================\n")

  train_data <- .positive_event_level(train_data, clase_positiva = clase_positiva)

  t_inicio_total <- Sys.time()
  tiempos_modelos <- list()
  metricas <- .make_metric_set()

  if (usar_cv == TRUE) {
    cat("  [!] Configuracion: Usando Cross-Validation (", folds, "-fold)\n", sep = "")
    set.seed(123)
    folds_obj <- rsample::vfold_cv(train_data, v = folds, strata = hospdead)
  } else {
    cat("  [!] Configuracion: Usando Entrenamiento Simple (Sin CV)\n")
    folds_obj <- NULL
  }

  specs <- .modelo_specs(scaled_df = scaled_df, incluir_svm = incluir_svm)
  modelos_entrenados <- list()

  for (nombre in names(specs)) {
    cat("\n------------------------------------------------------\n")
    cat(">> Entrenando Modelo:", nombre, "con tidymodels...\n")
    cat("------------------------------------------------------\n")

    t_inicio_mod <- Sys.time()

    modelos_entrenados[[nombre]] <- .fit_tidymodels_model(
      nombre = nombre,
      spec = specs[[nombre]]$spec,
      train_data = train_data,
      usar_cv = usar_cv,
      folds = folds_obj,
      metricas = metricas,
      grid_size = grid_size,
      tune_model = specs[[nombre]]$tune
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

  modelos_entrenados
}

.predict_modelo_tidymodels <- function(modelo, test_data, clase_positiva) {
  if (identical(modelo$engine, "tidymodels")) {
    probs <- stats::predict(modelo$fit, test_data, type = "prob")
    clases <- stats::predict(modelo$fit, test_data, type = "class")
    prob_col <- paste0(".pred_", clase_positiva)

    if (!prob_col %in% names(probs)) {
      stop("No se encontro la columna de probabilidad ", prob_col, ".", call. = FALSE)
    }

    return(data.frame(
      class_predictions = .extract_class_predictions(clases),
      positive_probability = probs[[prob_col]]
    ))
  }

  stop("Tipo de modelo no reconocido: ", modelo$engine, call. = FALSE)
}

.extract_class_predictions <- function(classes) {
  if (is.data.frame(classes) && ".pred_class" %in% names(classes)) {
    return(as.character(classes$.pred_class))
  }

  if (is.data.frame(classes)) {
    if (ncol(classes) == 1) {
      return(as.character(classes[[1]]))
    }
    stop("No se pudo interpretar la salida de prediccion de clase.", call. = FALSE)
  }

  if (is.factor(classes)) {
    return(as.character(classes))
  }

  if (is.character(classes)) {
    return(classes)
  }

  if (is.atomic(classes) && !is.list(classes)) {
    return(as.character(classes))
  }

  stop("No se pudo interpretar la salida de prediccion de clase.", call. = FALSE)
}

evaluar_benchmark <- function(modelos_lista, test_data, nombre_dataset, clase_positiva = "Yes") {
  .check_required_packages(c("yardstick", "pROC"))

  cat("\n--- Evaluando:", nombre_dataset, "en el set de TEST ---\n")

  test_data <- .positive_event_level(test_data, clase_positiva = clase_positiva)
  resultados_finales <- data.frame()

  for (nombre in names(modelos_lista)) {
    modelo <- modelos_lista[[nombre]]
    pred <- .predict_modelo_tidymodels(modelo, test_data, clase_positiva)

    filas_validas <- !is.na(pred$class_predictions) & !is.na(pred$positive_probability)
    y_test_valid <- test_data$hospdead[filas_validas]
    probs_valid <- pred$positive_probability[filas_validas]
    clases_valid <- factor(
      pred$class_predictions[filas_validas],
      levels = levels(y_test_valid)
    )

    roc_obj <- .roc_for_positive_class(y_test_valid, probs_valid, clase_positiva)
    sens <- yardstick::sens_vec(y_test_valid, clases_valid, event_level = "first")
    spec <- yardstick::spec_vec(y_test_valid, clases_valid, event_level = "first")
    acc <- yardstick::accuracy_vec(y_test_valid, clases_valid)

    tp <- sum(y_test_valid == clase_positiva & clases_valid == clase_positiva, na.rm = TRUE)
    tn <- sum(y_test_valid != clase_positiva & clases_valid != clase_positiva, na.rm = TRUE)
    fp <- sum(y_test_valid != clase_positiva & clases_valid == clase_positiva, na.rm = TRUE)
    fn <- sum(y_test_valid == clase_positiva & clases_valid != clase_positiva, na.rm = TRUE)
    precision <- if ((tp + fp) == 0) 0 else tp / (tp + fp)
    recall <- if ((tp + fn) == 0) 0 else tp / (tp + fn)
    f1 <- if ((precision + recall) == 0) 0 else 2 * precision * recall / (precision + recall)

    fila <- data.frame(
      Dataset = nombre_dataset,
      Modelo = nombre,
      AUC = as.numeric(pROC::auc(roc_obj)),
      Sensibilidad = sens,
      Especificidad = spec,
      Accuracy = acc,
      `F1-score` = f1
    )

    resultados_finales <- rbind(resultados_finales, fila)
  }

  resultados_finales
}

evaluar_benchmark_valid_test <- function(modelos_lista, valid_data, test_data,
                                        nombre_dataset, clase_positiva = "Yes") {
  valid_res <- evaluar_benchmark(
    modelos_lista = modelos_lista,
    test_data = valid_data,
    nombre_dataset = paste0(nombre_dataset, "_Valid"),
    clase_positiva = clase_positiva
  )

  test_res <- evaluar_benchmark(
    modelos_lista = modelos_lista,
    test_data = test_data,
    nombre_dataset = paste0(nombre_dataset, "_Test"),
    clase_positiva = clase_positiva
  )

  list(
    validation = valid_res,
    test = test_res
  )
}

generar_graficos_automaticos <- function(resultados) {
  .check_required_packages(c("dplyr", "ggplot2", "plotly", "htmltools"))

  datasets_unicos <- unique(resultados$Dataset)
  metricas <- c("AUC", "Sensibilidad", "Especificidad", "Accuracy", "F1-score")
  lista_graficos <- list()

  for (ds in datasets_unicos) {
    datos_ds <- dplyr::filter(resultados, Dataset == ds)

    for (met in metricas) {
      modelo_max <- datos_ds$Modelo[which.max(datos_ds[[met]])]

      p <- ggplot2::ggplot(
        datos_ds,
        ggplot2::aes(
          x = reorder(Modelo, .data[[met]]),
          y = .data[[met]],
          fill = Modelo == modelo_max,
          text = paste("Modelo:", Modelo, "<br>", met, ":", round(.data[[met]], 3))
        )
      ) +
        ggplot2::geom_bar(stat = "identity", width = 0.6) +
        ggplot2::scale_fill_manual(values = c("TRUE" = "#abc4ff", "FALSE" = "#a8e69d")) +
        ggplot2::theme_minimal() +
        ggplot2::theme(
          plot.background = ggplot2::element_rect(fill = "#121212", color = NA),
          panel.background = ggplot2::element_rect(fill = "#121212", color = NA),
          text = ggplot2::element_text(color = "white"),
          axis.text = ggplot2::element_text(color = "gray80"),
          axis.text.x = ggplot2::element_text(size = 12, face = "bold"),
          panel.grid.major.y = ggplot2::element_line(color = "gray30"),
          panel.grid.minor = ggplot2::element_blank(),
          panel.grid.major.x = ggplot2::element_blank(),
          legend.position = "none",
          plot.title = ggplot2::element_text(size = 16, face = "bold", margin = ggplot2::margin(b = 5)),
          plot.subtitle = ggplot2::element_text(size = 12, color = "gray50", margin = ggplot2::margin(b = 15))
        ) +
        ggplot2::labs(
          title = paste("Comparativa de", met),
          subtitle = paste("Dataset:", ds),
          x = "",
          y = paste("Valor de", met, "↑")
        ) +
        ggplot2::coord_cartesian(ylim = c(0, 1))

      p_interactivo <- plotly::ggplotly(p, tooltip = "text") |>
        plotly::layout(plot_bgcolor = "#121212", paper_bgcolor = "#121212")

      lista_graficos[[paste(ds, met, sep = "_")]] <- p_interactivo
    }
  }

  htmltools::browsable(htmltools::tagList(lista_graficos))
}

plot_roc_comparativo <- function(modelos_lista, test_data, titulo = "Comparación de Curvas ROC",
                                 clase_positiva = "Yes") {
  .check_required_packages(c("pROC", "ggplot2"))

  test_data <- .positive_event_level(test_data, clase_positiva = clase_positiva)
  lista_roc <- list()

  for (nombre in names(modelos_lista)) {
    pred <- .predict_modelo_tidymodels(modelos_lista[[nombre]], test_data, clase_positiva)

    filas_validas <- !is.na(pred$.pred_positive)
    y_test_valid <- test_data$hospdead[filas_validas]
    prob_valid <- pred$.pred_positive[filas_validas]

    lista_roc[[nombre]] <- .roc_for_positive_class(y_test_valid, prob_valid, clase_positiva)
  }

  pROC::ggroc(lista_roc, legacy.axes = TRUE) +
    ggplot2::theme_minimal() +
    ggplot2::geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "grey") +
    ggplot2::labs(
      title = titulo,
      x = "Tasa de Falsos Positivos (1-Especificidad)",
      y = "Tasa de Verdaderos Positivos (Sensibilidad)"
    ) +
    ggplot2::theme(
      legend.title = ggplot2::element_blank(),
      plot.title = ggplot2::element_text(face = "bold", size = 14)
    )
}
