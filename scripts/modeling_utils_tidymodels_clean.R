.check_required_packages <- function(packages) {
  missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) > 0) {
    stop(
      "Faltan paquetes requeridos para modeling_utils_tidymodels_clean.R: ",
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
    stop("No se encontro la variable objetivo '", target, "'.", call. = FALSE)
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

  tibble::tibble(
    AUC = as.numeric(pROC::auc(roc_obj)),
    Sensibilidad = yardstick::sens_vec(truth, estimate, event_level = "first"),
    Especificidad = yardstick::spec_vec(truth, estimate, event_level = "first"),
    Accuracy = yardstick::accuracy_vec(truth, estimate)
  )
}

.extract_prob_column <- function(probs, positive) {
  prob_col <- paste0(".pred_", positive)
  if (!prob_col %in% names(probs)) {
    stop("No se encontro la columna de probabilidad ", prob_col, ".", call. = FALSE)
  }
  probs[[prob_col]]
}

.make_recipe <- function(train_data, target = "hospdead") {
  recipes::recipe(stats::as.formula(paste(target, "~ .")), data = train_data) |>
    recipes::step_novel(recipes::all_nominal_predictors()) |>
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
    GBM = list(
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
    SVM = list(
      spec = parsnip::svm_rbf(
        cost = tune::tune(),
        rbf_sigma = tune::tune()
      ) |>
        parsnip::set_engine("kernlab") |>
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
    GBM = dials::grid_space_filling(
      dials::trees(range = c(200L, 700L)),
      dials::tree_depth(range = c(2L, 8L)),
      dials::learn_rate(range = c(-4, -1)),
      dials::loss_reduction(range = c(-6, 1)),
      dials::min_n(range = c(2L, 20L)),
      size = grid_size
    ),
    SVM = dials::grid_space_filling(
      dials::cost(range = c(-4, 2)),
      dials::rbf_sigma(range = c(-6, -1)),
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
      estimate <- factor(classes$.pred_class, levels = levels(truth))
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
          error = conditionMessage(e)
        )
      )
    })

    res
  })

  scored <- scored |>
    dplyr::arrange(dplyr::desc(AUC), dplyr::desc(Accuracy), dplyr::desc(Sensibilidad), dplyr::desc(Especificidad))

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
    metrics = .metrics_spec(),
    control = tune::control_grid(save_pred = TRUE, verbose = FALSE)
  )

  best_params <- tune::select_best(cv_res, metric = "roc_auc")
  final_wf <- tune::finalize_workflow(workflow_obj, best_params)
  fit_obj <- workflows::fit(final_wf, data = train_data)

  list(
    tuning = cv_res,
    best_params = best_params,
    fit_train = fit_obj
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

.fit_single_model <- function(model_name, model_spec, recipe_obj, train_data, valid_data, usar_cv, folds, grid_size, positive) {
  workflow_obj <- workflows::workflow() |>
    workflows::add_recipe(recipe_obj) |>
    workflows::add_model(model_spec)

  if (usar_cv) {
    set.seed(123)
    folds_obj <- rsample::vfold_cv(train_data, v = folds, strata = hospdead)
    tuned <- .tune_with_cv(model_name, workflow_obj, train_data, folds_obj, grid_size, positive)
  } else {
    tuned <- .tune_with_validation(model_name, workflow_obj, train_data, valid_data, grid_size, positive, recipe_obj)
  }

  valid_eval <- .prepare_target(valid_data, positive = positive)
  valid_probs <- stats::predict(tuned$fit_train, new_data = valid_eval, type = "prob")
  valid_classes <- stats::predict(tuned$fit_train, new_data = valid_eval, type = "class")
  valid_truth <- valid_eval$hospdead
  valid_estimate <- factor(valid_classes$.pred_class, levels = levels(valid_truth))
  valid_prob_positive <- .extract_prob_column(valid_probs, positive)
  valid_metrics <- .evaluate_predictions(valid_truth, valid_estimate, valid_prob_positive, positive) |>
    dplyr::mutate(Modelo = model_name, .before = 1)

  list(
    modelo = model_name,
    fit_train = tuned$fit_train,
    best_params = tuned$best_params,
    tuning = tuned$tuning,
    valid_metrics = valid_metrics,
    valid_predictions = tibble::tibble(
      truth = valid_truth,
      estimate = valid_estimate,
      .pred_positive = valid_prob_positive
    )
  )
}

entrenar_benchmark <- function(train_data, valid_data, test_data, nombre_dataset,
                               usar_cv = TRUE, folds = 5, grid_size = 10,
                               clase_positiva = "Yes", modelos = c("LogisticRegression", "DecisionTree", "RandomForest", "GBM", "SVM", "KNN")) {
  .check_required_packages(c(
    "recipes", "workflows", "tune", "rsample", "yardstick", "parsnip",
    "dials", "dplyr", "tibble", "purrr", "ggplot2", "plotly", "htmltools",
    "pROC", "glmnet", "ranger", "xgboost", "kernlab", "kknn", "rpart"
  ))

  cat("\n======================================================\n")
  cat("--- Iniciando benchmark tidymodels limpio para:", nombre_dataset, "---\n")
  cat("======================================================\n")

  set.seed(123)
  train_data <- .prepare_target(train_data, positive = clase_positiva)
  valid_data <- .prepare_target(valid_data, positive = clase_positiva)
  test_data <- .prepare_target(test_data, positive = clase_positiva)

  recipe_obj <- .make_recipe(train_data)
  catalog <- .model_catalog()
  modelos <- intersect(modelos, names(catalog))

  if (length(modelos) == 0) {
    stop("No se selecciono ningun modelo valido.", call. = FALSE)
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
    dplyr::arrange(dplyr::desc(AUC), dplyr::desc(Accuracy), dplyr::desc(Sensibilidad), dplyr::desc(Especificidad))

  best_row <- validacion_tbl |>
    dplyr::slice(1)

  best_model_name <- best_row$Modelo[[1]]
  best_params <- detalles_modelos[[best_model_name]]$best_params

  cat("\n======================================================\n")
  cat("Mejor modelo en validacion:", best_model_name, "\n")
  cat("======================================================\n")

  train_valid_data <- dplyr::bind_rows(train_data, valid_data)
  best_workflow <- workflows::workflow() |>
    workflows::add_recipe(recipe_obj) |>
    workflows::add_model(catalog[[best_model_name]]$spec)

  best_workflow <- tune::finalize_workflow(best_workflow, best_params)
  best_final_fit <- workflows::fit(best_workflow, data = .prepare_target(train_valid_data, positive = clase_positiva))

  test_eval <- .prepare_target(test_data, positive = clase_positiva)
  test_probs <- stats::predict(best_final_fit, new_data = test_eval, type = "prob")
  test_classes <- stats::predict(best_final_fit, new_data = test_eval, type = "class")
  test_truth <- test_eval$hospdead
  test_estimate <- factor(test_classes$.pred_class, levels = levels(test_truth))
  test_prob_positive <- .extract_prob_column(test_probs, clase_positiva)
  test_metrics <- .evaluate_predictions(test_truth, test_estimate, test_prob_positive, clase_positiva) |>
    dplyr::mutate(Modelo = best_model_name, Dataset = nombre_dataset, .before = 1)

  list(
    dataset = nombre_dataset,
    usar_cv = usar_cv,
    validation = validacion_tbl,
    test = test_metrics,
    best_model = best_model_name,
    best_params = best_params,
    best_fit = best_final_fit,
    validation_fits = fitting_train,
    model_details = detalles_modelos
  )
}

generar_graficos_automaticos <- function(resultados) {
  datasets_unicos <- unique(resultados$Dataset)
  metricas <- c("AUC", "Sensibilidad", "Especificidad", "Accuracy")
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
    estimate <- factor(clases$.pred_class, levels = levels(truth))
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
