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

# ==============================================================================
# IMAGEM 1 (SLIDE 3): Apenas o efeito da maldição (sem PCA)
# ==============================================================================
png("slide3_maldicao.png", width = 1200, height = 420, res = 140)
par(mfrow = c(1, 3), mar = c(4.5, 4, 3, 1.5))

graf_s3 <- function(col, titulo) {
  s <- tab[tab$versao == "original", ]
  plot(s$d, s[[col]], type = "b", pch = 16, lty = 1, lwd = 2, col = "#b91c1c",
       log = "x", xlab = "dimensão d", ylab = "", main = titulo,
       cex.main = 1.15, cex.lab = 1.05)
}

graf_s3("contraste", "1. Contraste relativo")
graf_s3("K_medio",   "2. K médio fora da diagonal")
graf_s3("nSV",       "3. Nº de vetores de suporte (de 150)")
dev.off()

# ==============================================================================
# IMAGEM 2 (SLIDE 5): Comparação Original vs. PCA + Dispersão das 2 PCs
# ==============================================================================
png("slide5_pca.png", width = 1200, height = 650, res = 140)
par(mfrow = c(2, 2), mar = c(4.2, 4, 2.8, 1.5))

graf_s5 <- function(col, titulo, incluir_legenda = FALSE) {
  plot(NA, xlim = range(tab$d), ylim = range(tab[[col]], na.rm = TRUE),
       log = "x", xlab = "dimensão d", ylab = "", main = titulo,
       cex.main = 1.1, cex.lab = 1.0)
  for (v in c("original", "PCA")) {
    s <- tab[tab$versao == v, ]
    lines(s$d, s[[col]], type = "b", lwd = 2,
          col = ifelse(v == "PCA", "#1d4ed8", "#64748b"),
          pch = ifelse(v == "PCA", 17, 16),
          lty = ifelse(v == "PCA", 1, 2))
  }
  if (incluir_legenda) {
    legend("topright", c("Original (com ruído)", "Após PCA (2 comp.)"),
           col = c("#64748b", "#1d4ed8"), pch = c(16, 17), lty = c(2, 1),
           lwd = 2, bty = "n", cex = 0.95)
  }
}

graf_s5("contraste", "Contraste relativo: Original vs. PCA", incluir_legenda = TRUE)
graf_s5("K_medio",   "K médio fora da diagonal: Original vs. PCA")
graf_s5("nSV",       "Vetores de suporte (de 150): Original vs. PCA")

Xg <- cbind(X0, matrix(rnorm(n * max(extras), sd = sd_ruido), n, max(extras)))
Zg <- prcomp(Xg)$x[, 1:2]
cores <- c("#16a34a", "#e11d48", "#0f172a")[as.numeric(iris$Species)]
plot(Zg, col = cores, pch = 19, xlab = "PC1", ylab = "PC2",
     main = paste0("Separação recuperada nas 2 PCs (d = ", ncol(Xg), ")"),
     cex.main = 1.1, cex.lab = 1.0)
legend("topright", levels(iris$Species), col = c("#16a34a", "#e11d48", "#0f172a"),
       pch = 19, bty = "n", cex = 0.9)
dev.off()
par(mfrow = c(1, 1))