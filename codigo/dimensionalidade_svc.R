library(kernlab)
library(mclust)   # adjustedRandIndex

set.seed(42)
X0 <- scale(iris[, 1:4]); n <- nrow(X0)
q0 <- 1 / median(dist(X0)^2)      # q fixo, calibrado no iris original (d = 4)

sd_ruido <- 0.5                   # desvio-padrão do ruído (pode mudar)
extras   <- c(0, 4, 16, 46, 96, 196)
k_pca    <- 2
nrep     <- 10

contraste <- function(X) {
  d <- as.numeric(dist(X))
  sd(d) / mean(d)
}

metricas <- function(X, q) {
  D2 <- as.matrix(dist(X))^2
  Kv <- exp(-q * D2)[upper.tri(D2)]
  f  <- ksvm(X, type = "one-svc", kernel = "rbfdot",
             kpar = list(sigma = q), nu = 0.2, scaled = FALSE)
  km <- kmeans(X, 3, nstart = 20)
  c(contraste = contraste(X), K_medio = mean(Kv), K_dp = sd(Kv),
    nSV = length(SVindex(f)),
    ARI = adjustedRandIndex(km$cluster, iris$Species))
}

res <- do.call(rbind, lapply(extras, function(e) {
  do.call(rbind, lapply(1:nrep, function(r) {
    X  <- if (e == 0) X0 else cbind(X0, matrix(rnorm(n * e, sd = sd_ruido), n, e))
    pc <- prcomp(X, center = TRUE, scale. = FALSE)
    Z  <- pc$x[, 1:k_pca]
    rbind(
      data.frame(d = ncol(X), versao = "original", t(metricas(X, q0)),
                 sinal_nas_PCs = NA),
      data.frame(d = ncol(X), versao = "PCA", t(metricas(Z, q0)),
                 sinal_nas_PCs = sum(pc$rotation[1:4, 1:k_pca]^2) / k_pca)
    )
  }))
}))

tab <- aggregate(. ~ d + versao, data = res, FUN = mean, na.action = na.pass)
print(tab[order(tab$versao, tab$d), ], digits = 3)

graf <- function(col, titulo, ylog = FALSE) {
  plot(NA, xlim = range(tab$d), ylim = range(tab[[col]], na.rm = TRUE),
       log = if (ylog) "xy" else "x", xlab = "dimensão d", ylab = "", main = titulo)
  for (v in c("original", "PCA")) {
    s <- tab[tab$versao == v, ]
    lines(s$d, s[[col]], type = "b",
          pch = ifelse(v == "PCA", 17, 16), lty = ifelse(v == "PCA", 2, 1))
  }
}
par(mfrow = c(2, 3))
graf("contraste", "Contraste relativo")
graf("K_medio",   "K médio fora da diagonal")
graf("nSV",       "Nº de vetores de suporte (de 150)")
graf("ARI",       "ARI do k-means")
plot(NA, xlim = c(0, 1), ylim = c(0, 1), axes = FALSE, xlab = "", ylab = "")
legend("center", c("original", "PCA (2 comp.)"), pch = c(16, 17), lty = c(1, 2))
Xg <- cbind(X0, matrix(rnorm(n * max(extras), sd = sd_ruido), n, max(extras)))
Zg <- prcomp(Xg)$x[, 1:2]
plot(Zg, col = iris$Species, pch = 19, main = paste0("2 PCs, d = ", ncol(Xg)))
par(mfrow = c(1, 1))