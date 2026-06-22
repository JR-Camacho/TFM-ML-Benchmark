.check_required_packages <- function(packages) {
  missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) > 0) {
    stop(
      "Faltan paquetes requeridos para modeling_utils_tidymodels.R: ",
      paste(missing, collapse = ", "),
      ". Instala antes de entrenar con install.packages(c(\"",
      paste(missing, collapse = "\", \""),
      "\"))",
      call. = FALSE
    )
  }
}

.prepare_target <- function(data, target = "hospdead", positive = "Yes") {
  if (!target %in% names(data)) {
    stop("No se encontró la variable objetivo '", target, "'.", call. = FALSE)
  }

  if (!is.factor(data[[target]])) {
    data[[target]] <- as.factor(data[[target]])
  }

  if (!positive %in% levels(data[[target]])) {
    stop(
      "La clase positiva '", positive, "' no existe en los niveles de ",
      target, ": ", paste(levels(data[[target]]), collapse = ", "),
      call. = FALSE
    )
  }

  data[[target]] <- stats::relevel(data[[target]], ref = positive)
  data
}

.metrics_spec <- function() {
  yardstick::metric_set(
    yardstick::roc_auc,
    yardstick::sens,
    yardstick::spec,
    yardstick::accuracy
  )
}

.cv_metrics_spec <- function(model_name) {
  .metrics_spec()
}

.cv_validation_metrics <- function(cv_res, train_data, positive, target = "hospdead") {
  pred_tbl <- tune::collect_predictions(cv_res)
  metric_tbl <- tune::collect_metrics(cv_res)

  if (!".config" %in% names(pred_tbl)) {
    stop("No se pudieron recuperar las predicciones de validación cruzada.", call. = FALSE)
  }

  best_config <- metric_tbl |>
    dplyr::filter(.metric == "roc_auc") |>
    dplyr::arrange(dplyr::desc(mean), dplyr::desc(std_err)) |>
    dplyr::slice(1) |>
    dplyr::pull(.config)

  best_preds <- dplyr::filter(pred_tbl, .config == best_config)
  truth_col <- intersect(c(".obs", ".truth", target), names(best_preds))[1]
  if (is.na(truth_col)) {
    stop("No se pudo identificar la columna de verdad en CV.", call. = FALSE)
  }

  truth_ref <- .prepare_target(train_data, target = target, positive = positive)[[target]]
  truth <- factor(best_preds[[truth_col]], levels = levels(truth_ref))
  estimate <- factor(.extract_class_predictions(best_preds), levels = levels(truth))
  prob_positive <- .extract_prob_column(best_preds, positive)

  .evaluate_predictions(truth, estimate, prob_positive, positive)
}

.roc_for_positive_class <- function(truth, prob_positive, positive) {
  negative <- setdiff(levels(truth), positive)
  if (length(negative) != 1) {
    stop(
      "Se esperaba una clasificacion binaria. Niveles: ",
      paste(levels(truth), collapse = ", "),
      call. = FALSE
    )
  }

  pROC::roc(
    truth,
    prob_positive,
    levels = c(negative, positive),
    quiet = TRUE
  )
}

.evaluate_predictions <- function(truth, estimate, prob_positive, positive) {
  roc_obj <- .roc_for_positive_class(truth, prob_positive, positive)

  truth_chr <- as.character(truth)
  estimate_chr <- as.character(estimate)
  positive_chr <- as.character(positive)

  tp <- sum(truth_chr == positive_chr & estimate_chr == positive_chr, na.rm = TRUE)
  tn <- sum(truth_chr != positive_chr & estimate_chr != positive_chr, na.rm = TRUE)
  fp <- sum(truth_chr != positive_chr & estimate_chr == positive_chr, na.rm = TRUE)
  fn <- sum(truth_chr == positive_chr & estimate_chr != positive_chr, na.rm = TRUE)

  precision <- if ((tp + fp) == 0) 0 else tp / (tp + fp)
  recall <- if ((tp + fn) == 0) 0 else tp / (tp + fn)
  specificity <- if ((tn + fp) == 0) 0 else tn / (tn + fp)
  accuracy <- if ((tp + tn + fp + fn) == 0) 0 else (tp + tn) / (tp + tn + fp + fn)
  f1 <- if ((precision + recall) == 0) 0 else 2 * precision * recall / (precision + recall)

  tibble::tibble(
    AUC = as.numeric(pROC::auc(roc_obj)),
    Sensibilidad = recall,
    Especificidad = specificity,
    Accuracy = accuracy,
    `F1-score` = f1
  )
}

.score_model_predictions <- function(fit, data, positive, target = "hospdead") {
  prepared <- .prepare_target(data, target = target, positive = positive)
  probs <- stats::predict(fit, new_data = prepared, type = "prob")
  classes <- stats::predict(fit, new_data = prepared, type = "class")

  truth <- prepared[[target]]
  estimate <- factor(.extract_class_predictions(classes), levels = levels(truth))
  prob_positive <- .extract_prob_column(probs, positive)

  list(
    truth = truth,
    estimate = estimate,
    prob_positive = prob_positive
  )
}

.permutation_importance <- function(fit, data, positive, target = "hospdead",
                                    n_repeats = 5, seed = 123) {
  .check_required_packages(c("dplyr", "purrr", "tibble"))

  if (n_repeats < 1) {
    stop("n_repeats debe ser al menos 1.", call. = FALSE)
  }

  scored_base <- .score_model_predictions(fit, data, positive, target = target)
  baseline_metrics <- .evaluate_predictions(
    scored_base$truth,
    scored_base$estimate,
    scored_base$prob_positive,
    positive
  )
  baseline_auc <- baseline_metrics$AUC[[1]]

  predictor_names <- setdiff(names(.prepare_target(data, target = target, positive = positive)), target)
  set.seed(seed)

  importancia_tbl <- purrr::map_dfr(predictor_names, function(variable) {
    deltas <- replicate(n_repeats, {
      permuted_data <- data
      permuted_data[[variable]] <- permuted_data[[variable]][sample.int(nrow(permuted_data))]

      permuted_scored <- .score_model_predictions(fit, permuted_data, positive, target = target)
      permuted_metrics <- .evaluate_predictions(
        permuted_scored$truth,
        permuted_scored$estimate,
        permuted_scored$prob_positive,
        positive
      )

      baseline_auc - permuted_metrics$AUC[[1]]
    })

    tibble::tibble(
      Variable = variable,
      AUC_base = baseline_auc,
      AUC_perm_media = baseline_auc - mean(deltas, na.rm = TRUE),
      Importancia = mean(deltas, na.rm = TRUE),
      Importancia_sd = stats::sd(deltas)
    )
  })

  dplyr::arrange(importancia_tbl, dplyr::desc(Importancia), Variable)
}

.extract_prob_column <- function(probs, positive) {
  prob_col <- paste0(".pred_", positive)
  if (!prob_col %in% names(probs)) {
    stop("No se encontró la columna de probabilidad ", prob_col, ".", call. = FALSE)
  }
  probs[[prob_col]]
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

.make_recipe <- function(train_data, target = "hospdead", usar_downsampling = FALSE) {
  recipe_obj <- recipes::recipe(stats::as.formula(paste(target, "~ .")), data = train_data) |>
    recipes::step_novel(recipes::all_nominal_predictors())

  if (usar_downsampling) {
    recipe_obj <- recipe_obj |>
      themis::step_downsample(recipes::all_outcomes())
  }

  recipe_obj |>
    recipes::step_normalize(recipes::all_numeric_predictors()) |>
    recipes::step_dummy(recipes::all_nominal_predictors(), one_hot = TRUE) |>
    recipes::step_zv(recipes::all_predictors())
}

.model_catalog <- function() {
  list(
    LogisticRegression = list(
      spec = parsnip::logistic_reg(
        penalty = tune::tune(),
        mixture = tune::tune()
      ) |>
        parsnip::set_engine("glmnet"),
      tunable = TRUE
    ),
    DecisionTree = list(
      spec = parsnip::decision_tree(
        cost_complexity = tune::tune(),
        tree_depth = tune::tune(),
        min_n = tune::tune()
      ) |>
        parsnip::set_engine("rpart") |>
        parsnip::set_mode("classification"),
      tunable = TRUE
    ),
    RandomForest = list(
      spec = parsnip::rand_forest(
        mtry = tune::tune(),
        trees = 500,
        min_n = tune::tune()
      ) |>
        parsnip::set_engine("ranger", importance = "impurity", probability = TRUE) |>
        parsnip::set_mode("classification"),
      tunable = TRUE
    ),
    NaiveBayes = list(
      spec = parsnip::naive_Bayes(
        smoothness = tune::tune(),
        Laplace = tune::tune()
      ) |>
        parsnip::set_engine("klaR") |>
        parsnip::set_mode("classification"),
      tunable = TRUE
    ),
    XGBoost = list(
      spec = parsnip::boost_tree(
        trees = tune::tune(),
        tree_depth = tune::tune(),
        learn_rate = tune::tune(),
        loss_reduction = tune::tune(),
        min_n = tune::tune()
      ) |>
        parsnip::set_engine("xgboost") |>
        parsnip::set_mode("classification"),
      tunable = TRUE
    ),
    KNN = list(
      spec = parsnip::nearest_neighbor(
        neighbors = tune::tune(),
        weight_func = tune::tune(),
        dist_power = tune::tune()
      ) |>
        parsnip::set_engine("kknn") |>
        parsnip::set_mode("classification"),
      tunable = TRUE
    )
  )
}

.manual_grid_knn <- function(train_data, recipe_obj, grid_size) {
  baked <- recipes::prep(recipe_obj, training = train_data, retain = TRUE) |>
    recipes::bake(new_data = train_data)
  p <- max(1L, ncol(baked) - 1L)
  max_neighbors <- max(3L, min(25L, nrow(train_data) - 1L, p))

  grid <- tidyr::crossing(
    neighbors = unique(pmax(1L, round(seq(3, max_neighbors, length.out = min(grid_size, 6L))))),
    weight_func = c("rectangular", "triangular", "epanechnikov"),
    dist_power = c(1, 2)
  )

  if (nrow(grid) > grid_size) {
    set.seed(123)
    grid <- dplyr::slice_sample(grid, n = grid_size)
  }

  grid
}

.manual_grid_other <- function(model_name, train_data, recipe_obj, grid_size) {
  baked <- recipes::prep(recipe_obj, training = train_data, retain = TRUE) |>
    recipes::bake(new_data = train_data)
  p <- max(1L, ncol(baked) - 1L)

  grid <- switch(
    model_name,
    LogisticRegression = dials::grid_space_filling(
      dials::penalty(range = c(-6, 0)),
      dials::mixture(range = c(0, 1)),
      size = grid_size
    ),
    DecisionTree = dials::grid_space_filling(
      dials::cost_complexity(range = c(-6, -1)),
      dials::tree_depth(range = c(1L, 15L)),
      dials::min_n(range = c(2L, 20L)),
      size = grid_size
    ),
    RandomForest = dials::grid_space_filling(
      dials::mtry(range = c(1L, p)),
      dials::min_n(range = c(2L, 20L)),
      size = grid_size
    ),
    NaiveBayes = dials::grid_space_filling(
      dials::smoothness(range = c(0.5, 1.5)),
      dials::Laplace(range = c(0, 3)),
      size = grid_size
    ),
    XGBoost = dials::grid_space_filling(
      dials::trees(range = c(200L, 700L)),
      dials::tree_depth(range = c(2L, 8L)),
      dials::learn_rate(range = c(-4, -1)),
      dials::loss_reduction(range = c(-6, 1)),
      dials::min_n(range = c(2L, 20L)),
      size = grid_size
    ),
    NULL
  )

  if (is.null(grid)) {
    stop("No hay grilla manual definida para ", model_name, ".", call. = FALSE)
  }

  grid
}

.fit_and_score_grid <- function(model_name, workflow_obj, train_data, valid_data, grid, positive) {
  base_cols <- names(grid)

  scored <- purrr::map_dfr(seq_len(nrow(grid)), function(i) {
    params <- grid[i, , drop = FALSE]

    res <- tryCatch({
      final_wf <- tune::finalize_workflow(workflow_obj, params)
      fit_obj <- workflows::fit(final_wf, data = train_data)
      probs <- stats::predict(fit_obj, new_data = valid_data, type = "prob")
      classes <- stats::predict(fit_obj, new_data = valid_data, type = "class")
      truth <- .prepare_target(valid_data, positive = positive)$hospdead
      estimate <- factor(.extract_class_predictions(classes), levels = levels(truth))
      prob_positive <- .extract_prob_column(probs, positive)
      metrics <- .evaluate_predictions(truth, estimate, prob_positive, positive)

      dplyr::bind_cols(
        tibble::tibble(.row = i),
        params,
        metrics
      )
    }, error = function(e) {
      dplyr::bind_cols(
        tibble::tibble(.row = i),
        params,
        tibble::tibble(
          AUC = NA_real_,
          Sensibilidad = NA_real_,
          Especificidad = NA_real_,
          Accuracy = NA_real_,
          `F1-score` = NA_real_,
          error = conditionMessage(e)
        )
      )
    })

    res
  })

  scored <- scored |>
    dplyr::arrange(
      dplyr::desc(AUC),
      dplyr::desc(`F1-score`),
      dplyr::desc(Accuracy),
      dplyr::desc(Sensibilidad),
      dplyr::desc(Especificidad)
    )

  best_params <- scored |>
    dplyr::slice(1) |>
    dplyr::select(dplyr::all_of(base_cols))

  list(scored = scored, best_params = best_params)
}

.tune_with_cv <- function(model_name, workflow_obj, train_data, folds, grid_size, positive) {
  cv_res <- tune::tune_grid(
    workflow_obj,
    resamples = folds,
    grid = grid_size,
    metrics = .cv_metrics_spec(model_name),
    control = tune::control_grid(save_pred = TRUE, verbose = FALSE)
  )

  best_params <- tune::select_best(cv_res, metric = "roc_auc")
  final_wf <- tune::finalize_workflow(workflow_obj, best_params)
  fit_obj <- workflows::fit(final_wf, data = train_data)
  cv_metrics <- .cv_validation_metrics(cv_res, train_data, positive)

  list(
    tuning = cv_res,
    best_params = best_params,
    fit_train = fit_obj,
    cv_metrics = cv_metrics
  )
}

.tune_with_validation <- function(model_name, workflow_obj, train_data, valid_data, grid_size, positive, recipe_obj) {
  grid <- if (model_name == "KNN") {
    .manual_grid_knn(train_data, recipe_obj, grid_size)
  } else {
    .manual_grid_other(model_name, train_data, recipe_obj, grid_size)
  }

  scored <- .fit_and_score_grid(model_name, workflow_obj, train_data, valid_data, grid, positive)
  best_params <- scored$best_params
  final_wf <- tune::finalize_workflow(workflow_obj, best_params)
  fit_obj <- workflows::fit(final_wf, data = train_data)

  list(
    tuning = scored$scored,
    best_params = best_params,
    fit_train = fit_obj
  )
}

.fit_single_model <- function(model_name, model_spec, recipe_obj, train_data, valid_data = NULL, usar_cv, folds, grid_size, positive) {
  workflow_obj <- workflows::workflow() |>
    workflows::add_recipe(recipe_obj) |>
    workflows::add_model(model_spec)

  if (usar_cv) {
    set.seed(123)
    folds_obj <- rsample::vfold_cv(train_data, v = folds, strata = hospdead)
    tuned <- .tune_with_cv(model_name, workflow_obj, train_data, folds_obj, grid_size, positive)
  } else {
    if (is.null(valid_data)) {
      stop("valid_data es obligatorio cuando usar_cv = FALSE.", call. = FALSE)
    }
    tuned <- .tune_with_validation(model_name, workflow_obj, train_data, valid_data, grid_size, positive, recipe_obj)
  }

  if (usar_cv) {
    valid_metrics <- tuned$cv_metrics |>
      dplyr::mutate(Modelo = model_name, .before = 1)
    valid_predictions <- NULL
  } else {
    valid_eval <- .prepare_target(valid_data, positive = positive)
    valid_probs <- stats::predict(tuned$fit_train, new_data = valid_eval, type = "prob")
    valid_classes <- stats::predict(tuned$fit_train, new_data = valid_eval, type = "class")
    valid_truth <- valid_eval$hospdead
    valid_estimate <- factor(.extract_class_predictions(valid_classes), levels = levels(valid_truth))
    valid_prob_positive <- .extract_prob_column(valid_probs, positive)
    valid_metrics <- .evaluate_predictions(valid_truth, valid_estimate, valid_prob_positive, positive) |>
      dplyr::mutate(Modelo = model_name, .before = 1)
    valid_predictions <- tibble::tibble(
      truth = valid_truth,
      estimate = valid_estimate,
      .pred_positive = valid_prob_positive
    )
  }

  list(
    modelo = model_name,
    fit_train = tuned$fit_train,
    best_params = tuned$best_params,
    tuning = tuned$tuning,
    valid_metrics = valid_metrics,
    valid_predictions = valid_predictions
  )
}

entrenar_benchmark <- function(train_data, valid_data = NULL, test_data = NULL, nombre_dataset,
                               usar_cv = TRUE, folds = 5, grid_size = 10,
                               usar_downsampling = FALSE,
                               clase_positiva = "Yes", modelos = c("LogisticRegression", "DecisionTree", "RandomForest", "NaiveBayes", "XGBoost", "KNN")) {
  .check_required_packages(c(
    "recipes", "workflows", "tune", "rsample", "yardstick", "parsnip",
    "dials", "dplyr", "tibble", "purrr", "ggplot2", "plotly", "htmltools",
    "pROC", "glmnet", "ranger", "kknn", "rpart", "discrim", "klaR"
  ))

  cat("\n======================================================\n")
  cat("--- Iniciando benchmark tidymodels limpio para:", nombre_dataset, "---\n")
  cat("======================================================\n")

  set.seed(123)
  train_data <- .prepare_target(train_data, positive = clase_positiva)
  if (usar_downsampling) {
    .check_required_packages(c("themis"))
    if (!usar_cv) {
      stop("usar_downsampling solo se permite cuando usar_cv = TRUE.", call. = FALSE)
    }
  }
  if (usar_cv) {
    if (is.null(test_data)) {
      stop("test_data es obligatorio cuando usar_cv = TRUE.", call. = FALSE)
    }
    test_data <- .prepare_target(test_data, positive = clase_positiva)
  } else {
    if (is.null(valid_data) || is.null(test_data)) {
      stop("valid_data y test_data son obligatorios cuando usar_cv = FALSE.", call. = FALSE)
    }
    valid_data <- .prepare_target(valid_data, positive = clase_positiva)
    test_data <- .prepare_target(test_data, positive = clase_positiva)
  }

  recipe_obj <- .make_recipe(train_data, usar_downsampling = usar_downsampling)
  catalog <- .model_catalog()
  invalid_models <- setdiff(modelos, names(catalog))
  if (length(invalid_models) > 0) {
    stop(
      "Los siguientes modelos no existen en el catalogo tidymodels limpio: ",
      paste(invalid_models, collapse = ", "),
      call. = FALSE
    )
  }

  if (length(modelos) == 0) {
    stop("No se seleccionó ningún modelo válido.", call. = FALSE)
  }

  detalles_modelos <- list()
  resumen_validacion <- list()
  fitting_train <- list()

  for (nombre in modelos) {
    cat("\n------------------------------------------------------\n")
    cat(">> Ajustando modelo:", nombre, "\n")
    cat("------------------------------------------------------\n")

    set.seed(123)
    ajuste <- .fit_single_model(
      model_name = nombre,
      model_spec = catalog[[nombre]]$spec,
      recipe_obj = recipe_obj,
      train_data = train_data,
      valid_data = valid_data,
      usar_cv = usar_cv,
      folds = folds,
      grid_size = grid_size,
      positive = clase_positiva
    )

    detalles_modelos[[nombre]] <- ajuste
    fitting_train[[nombre]] <- ajuste$fit_train
    resumen_validacion[[nombre]] <- ajuste$valid_metrics |>
      dplyr::mutate(Dataset = nombre_dataset, .before = 1)

    cat(">> Modelo listo:", nombre, "\n")
  }

  validacion_tbl <- dplyr::bind_rows(resumen_validacion) |>
    dplyr::arrange(
      dplyr::desc(AUC),
      dplyr::desc(`F1-score`),
      dplyr::desc(Accuracy),
      dplyr::desc(Sensibilidad),
      dplyr::desc(Especificidad)
    )

  best_row <- validacion_tbl |>
    dplyr::slice(1)

  best_model_name <- best_row$Modelo[[1]]
  best_params <- detalles_modelos[[best_model_name]]$best_params

  cat("\n======================================================\n")
  cat("Mejor modelo en validacion:", best_model_name, "\n")
  cat("======================================================\n")

  best_final_fit <- detalles_modelos[[best_model_name]]$fit_train

  importance_data <- if (is.null(valid_data)) train_data else valid_data
  variables_importantes <- .permutation_importance(
    fit = best_final_fit,
    data = importance_data,
    positive = clase_positiva,
    target = "hospdead",
    n_repeats = 5,
    seed = 123
  )

  test_results <- purrr::imap_dfr(fitting_train, function(fit_obj, nombre) {
    test_eval <- .prepare_target(test_data, positive = clase_positiva)
    test_probs <- stats::predict(fit_obj, new_data = test_eval, type = "prob")
    test_classes <- stats::predict(fit_obj, new_data = test_eval, type = "class")
    test_truth <- test_eval$hospdead
    test_estimate <- factor(.extract_class_predictions(test_classes), levels = levels(test_truth))
    test_prob_positive <- .extract_prob_column(test_probs, clase_positiva)

    .evaluate_predictions(test_truth, test_estimate, test_prob_positive, clase_positiva) |>
      dplyr::mutate(Modelo = nombre, Dataset = nombre_dataset, .before = 1)
  }) |>
    dplyr::arrange(
      dplyr::desc(AUC),
      dplyr::desc(`F1-score`),
      dplyr::desc(Accuracy),
      dplyr::desc(Sensibilidad),
      dplyr::desc(Especificidad)
    )

  best_test_metrics <- dplyr::filter(test_results, Modelo == best_model_name) |>
    dplyr::slice(1)

  list(
    dataset = nombre_dataset,
    usar_cv = usar_cv,
    validation = validacion_tbl,
    test = test_results,
    best_test = best_test_metrics,
    best_model = best_model_name,
    best_params = best_params,
    best_fit = best_final_fit,
    variable_importance = variables_importantes,
    validation_fits = fitting_train,
    model_details = detalles_modelos
  )
}

generar_graficos_automaticos <- function(resultados) {
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
        ggplot2::geom_col(width = 0.6) +
        ggplot2::scale_fill_manual(values = c("TRUE" = "#abc4ff", "FALSE" = "#a8e69d")) +
        ggplot2::theme_minimal() +
        ggplot2::theme(
          plot.background = ggplot2::element_rect(fill = "#121212", color = NA),
          panel.background = ggplot2::element_rect(fill = "#121212", color = NA),
          text = ggplot2::element_text(color = "white"),
          axis.text = ggplot2::element_text(color = "gray80"),
          axis.text.x = ggplot2::element_text(size = 11, face = "bold"),
          panel.grid.major.y = ggplot2::element_line(color = "gray30"),
          panel.grid.minor = ggplot2::element_blank(),
          panel.grid.major.x = ggplot2::element_blank(),
          legend.position = "none",
          plot.title = ggplot2::element_text(size = 15, face = "bold"),
          plot.subtitle = ggplot2::element_text(size = 11, color = "gray60")
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
  lista_roc <- list()
  test_data <- .prepare_target(test_data, positive = clase_positiva)

  for (nombre in names(modelos_lista)) {
    modelo <- modelos_lista[[nombre]]

    probs <- stats::predict(modelo, new_data = test_data, type = "prob")
    clases <- stats::predict(modelo, new_data = test_data, type = "class")

    truth <- test_data$hospdead
    estimate <- factor(.extract_class_predictions(clases), levels = levels(truth))
    prob_positive <- .extract_prob_column(probs, clase_positiva)

    lista_roc[[nombre]] <- .roc_for_positive_class(truth, prob_positive, clase_positiva)
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
