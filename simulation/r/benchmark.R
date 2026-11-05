# Portable FFM and TEV benchmark for the Python export format.
# Usage: Rscript simulation/r/benchmark.R --input-dir PATH/r_test --output results/r_metrics.csv
# Optional: --methods TEV,FFM --nbasis 9 --m 1000 --seed 2026 --sample-index 1
# Matrices are headerless (time, grid); labels and grid are headerless columns.

arguments <- commandArgs(trailingOnly=TRUE)
defaults <- list("input-dir"=NULL, output="results/r_metrics.csv", methods="TEV,FFM",
                 nbasis="9", m="1000", seed="2026", bandwidth="0", "sample-index"=NULL)
if (length(arguments) %% 2L != 0L) stop("Arguments must be --name value pairs.")
if (length(arguments)) {
  for (i in seq.int(1L, length(arguments), by=2L)) {
    key <- sub("^--", "", arguments[i])
    if (!key %in% names(defaults)) stop(paste("Unknown argument:", key))
    defaults[key] <- list(arguments[i + 1L])
  }
}
if (is.null(defaults[["input-dir"]])) stop("--input-dir is required.")
methods <- strsplit(defaults$methods, ",", fixed=TRUE)[[1L]]
if (any(!methods %in% c("TEV", "FFM"))) stop("Methods must be TEV and/or FFM.")
if (!requireNamespace("fda", quietly=TRUE)) stop("Install the R package 'fda'.")
if ("FFM" %in% methods) {
  if (!requireNamespace("fChange", quietly=TRUE) ||
      !"change_FF" %in% getNamespaceExports("fChange") ||
      utils::packageVersion("fChange") != "0.2.0") {
    stop(paste("FFM requires Ozan Sonmez's fChange 0.2.0 exporting change_FF.",
               "See README.md for installation, or use --methods TEV."))
  }
}
script_argument <- grep("^--file=", commandArgs(), value=TRUE)
script_directory <- dirname(normalizePath(sub("^--file=", "", script_argument[1L])))
source(file.path(script_directory, "tev.R"))

input <- defaults[["input-dir"]]
files <- list.files(input, pattern="^sample_[0-9]+\\.csv$", full.names=TRUE)
numeric_index <- as.integer(sub("^sample_([0-9]+)\\.csv$", "\\1", basename(files)))
files <- files[order(numeric_index)]
labels <- scan(file.path(input, "Y.csv"), sep=",", quiet=TRUE)
if (!length(files) || length(files) != length(labels) || any(!labels %in% c(0L, 1L))) {
  stop("Sample files and binary labels must be nonempty and have equal lengths.")
}
selected <- seq_along(files)
if (!is.null(defaults[["sample-index"]])) {
  selected <- as.integer(defaults[["sample-index"]])
  if (is.na(selected) || selected < 1L || selected > length(files)) stop("Invalid sample index.")
}
first <- as.matrix(read.csv(files[1L], header=FALSE))
grid_file <- file.path(input, "frequency_grid_hz.csv")
grid <- if (file.exists(grid_file)) scan(grid_file, sep=",", quiet=TRUE) else seq(0, 1, length.out=ncol(first))
if (length(grid) != ncol(first) || any(!is.finite(grid)) || any(diff(grid) <= 0)) {
  stop("The grid must be finite, strictly increasing, and match sample columns.")
}
nbasis <- as.integer(defaults$nbasis)
if (is.na(nbasis) || nbasis < 3L || nbasis > length(grid) || nbasis %% 2L == 0L) {
  stop("--nbasis must be odd, >=3, and no greater than the grid size.")
}
m <- as.integer(defaults$m)
seed <- as.integer(defaults$seed)
bandwidth <- as.integer(defaults$bandwidth)
if (any(is.na(c(m, seed, bandwidth))) || m < 1L || bandwidth < 0L) stop("Invalid M, seed, or bandwidth.")
basis <- fda::create.fourier.basis(rangeval=range(grid), nbasis=nbasis)
rows <- list()
errors <- list()
method_names <- c(if ("TEV" %in% methods) c("TEV-85", "TEV-90", "TEV-95"),
                  if ("FFM" %in% methods) "FFM")
for (index in selected) {
  sample <- tryCatch({
    matrix <- as.matrix(read.csv(files[index], header=FALSE))
    if (ncol(matrix) != length(grid) || nrow(matrix) < 3L || any(!is.finite(matrix))) {
      stop("Invalid functional sample.")
    }
    fda::Data2fd(argvals=grid, y=t(matrix), basisobj=basis)
  }, error=function(e) e)
  for (method in method_names) {
    result <- tryCatch({
      if (inherits(sample, "error")) stop(conditionMessage(sample))
      if (method == "FFM") {
        set.seed(as.integer((as.double(seed) * 100000 + index * 10 + 4) %% .Machine$integer.max))
        fit <- fChange::change_FF(sample, M=m, h=bandwidth, plot=FALSE)
        list(decision=as.integer(fit$pvalue <= 0.05), score=1 - fit$pvalue, change=fit$change)
      } else {
        fraction <- as.numeric(sub("TEV-", "", method)) / 100
        tev_test(sample, fraction)
      }
    }, error=function(e) e)
    if (inherits(result, "error")) {
      errors[[length(errors) + 1L]] <- data.frame(sample_index=index, method=method,
                                                message=conditionMessage(result))
      result <- list(decision=NA_real_, score=NA_real_, change=NA_real_)
    }
    rows[[length(rows) + 1L]] <- data.frame(sample_index=index, sample_file=basename(files[index]),
                                           label=labels[index], method=method,
                                           prediction=result$decision, score=result$score,
                                           change_location_1based=result$change)
  }
  if (index %% 25L == 0L) cat("Evaluated sample", index, "\n")
}
predictions <- do.call(rbind, rows)
metrics <- do.call(rbind, lapply(method_names, function(method) {
  subset <- predictions[predictions$method == method, ]
  cbind(method=method, binary_metrics(subset$label, subset$prediction, subset$score))
}))
output <- defaults$output
dir.create(dirname(output), recursive=TRUE, showWarnings=FALSE)
write.csv(metrics, output, row.names=FALSE)
write.csv(predictions, sub("(\\.csv)?$", "_predictions.csv", output), row.names=FALSE)
error_table <- if (length(errors)) do.call(rbind, errors) else data.frame(sample_index=integer(), method=character(), message=character())
write.csv(error_table, sub("(\\.csv)?$", "_errors.csv", output), row.names=FALSE)
print(metrics)
