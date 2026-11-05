# FPCA/TEV test equations and critical values used in the supplied comparison.
# Reference: Berkes, Gabrys, Horvath, and Kokoszka (2009), Table 1, alpha=0.05.
# This small implementation uses the installed fda package; no fChange source
# code is redistributed here.

tev_critical <- c(
  0.460496, 0.748785, 1.001390, 1.239675, 1.469008, 1.684729,
  1.895557, 2.124153, 2.322674, 2.526781, 2.744438, 2.949004,
  3.147604, 3.336262, 3.544633, 3.740248, 3.949054, 4.136169,
  4.327286, 4.532917, 4.718904, 4.908332, 5.101896, 5.303462,
  5.495721, 5.688849, 5.866095, 6.068351, 6.242770, 6.444772
)

tev_test <- function(fdobj, fraction) {
  max_harm <- min(nrow(fdobj$coefs), ncol(fdobj$coefs))
  decomposition <- fda::pca.fd(fdobj, nharm=max_harm)
  d <- which(cumsum(decomposition$varprop) >= fraction)[1L]
  if (is.na(d)) d <- length(decomposition$varprop)
  if (d > length(tev_critical)) stop("TEV critical values support at most 30 dimensions.")
  scores <- decomposition$scores[, seq_len(d), drop=FALSE]
  eigenvalues <- decomposition$values[seq_len(d)]
  if (any(!is.finite(eigenvalues)) || any(eigenvalues <= 0)) {
    stop("TEV requires positive retained eigenvalues.")
  }
  n <- nrow(scores)
  centered_sums <- apply(scores, 2L, cumsum) - outer(seq_len(n), colMeans(scores))
  centered_sums <- matrix(centered_sums, nrow=n, ncol=d)
  components <- sweep(centered_sums^2, 2L, eigenvalues, "/")
  statistic <- sum(components) / n^2
  list(decision=as.integer(statistic > tev_critical[d]),
       score=statistic, change=which.max(rowSums(components) / n), d=d)
}

binary_metrics <- function(y, prediction, score) {
  valid <- is.finite(prediction) & is.finite(score)
  total <- length(y)
  y <- y[valid]
  prediction <- prediction[valid]
  score <- score[valid]
  tp <- sum(y == 1L & prediction == 1L)
  fp <- sum(y == 0L & prediction == 1L)
  tn <- sum(y == 0L & prediction == 0L)
  fn <- sum(y == 1L & prediction == 0L)
  divide <- function(a, b) if (b == 0) NA_real_ else a / b
  positive <- sum(y == 1L)
  negative <- sum(y == 0L)
  auc <- if (positive == 0L || negative == 0L) NA_real_ else {
    (sum(rank(score, ties.method="average")[y == 1L]) - positive * (positive + 1) / 2) /
      (positive * negative)
  }
  data.frame(Accuracy=divide(tp + tn, length(y)), TPR=divide(tp, tp + fn),
             FAR=divide(fp, fp + tn), Precision=divide(tp, tp + fp),
             F1=divide(2 * tp, 2 * tp + fp + fn), AUC=auc,
             N=length(y), N_errors=total - length(y))
}
