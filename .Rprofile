local_lib <- file.path(getwd(), ".R", "library")

if (dir.exists(local_lib)) {
  .libPaths(unique(c(normalizePath(local_lib), .libPaths())))
}
