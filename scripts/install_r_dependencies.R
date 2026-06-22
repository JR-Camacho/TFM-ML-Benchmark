#!/usr/bin/env Rscript

get_script_path <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) == 1) {
    return(normalizePath(sub("^--file=", "", file_arg), mustWork = TRUE))
  }
  stop("Este script debe ejecutarse con Rscript.", call. = FALSE)
}

script_path <- get_script_path()
project_root <- normalizePath(file.path(dirname(script_path), ".."), mustWork = TRUE)
requirements_file <- file.path(project_root, "requirements-r.txt")
local_lib <- file.path(project_root, ".R", "library")

if (!file.exists(requirements_file)) {
  stop("No se encontro requirements-r.txt en la raiz del proyecto.", call. = FALSE)
}

dir.create(local_lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(normalizePath(local_lib), .libPaths()))
options(repos = c(CRAN = "https://cloud.r-project.org"))

packages <- readLines(requirements_file, warn = FALSE)
packages <- trimws(packages)
packages <- packages[nzchar(packages)]
packages <- packages[!startsWith(packages, "#")]

missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]

cat("Libreria local:", local_lib, "\n")
if (length(missing) == 0) {
  cat("Todas las dependencias de R ya estan instaladas.\n")
  quit(status = 0)
}

cat("Instalando paquetes faltantes:\n")
cat(paste0(" - ", missing, collapse = "\n"), "\n")

install.packages(missing, lib = local_lib)

cat("Instalacion completada.\n")
