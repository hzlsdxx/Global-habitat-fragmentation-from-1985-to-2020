# ==============================================================================
# glmm_fragmentation_analysis.R
# Gaussian GLMM with a random intercept for each 5-degree spatial block.
# Run from the repository root after configuring the input and output paths.
# Results are written as TXT; no figures or fitted model files are generated.
# ==============================================================================

library(terra)      # 栅格读写
library(glmmTMB)    # 空间 GLMM

# ==============================================================================
# 1. 全局配置（路径、参数）
# ==============================================================================

BLOCK_DEG    <- 5    # 经纬度分块大小；经向距离随纬度变化，不是等面积网格
MIN_BLOCK_N  <- 5    # block 内最少像元数，低于此值的 block 被剔除
# 注：全球区域像元数极多，GLMM 拟合很慢；如需加速可将下面设为较小值（如 300000）
MAX_OBS      <- 1000000  # 每个区域最大样本数（随机抽样，set.seed=42 可复现）

# ---- 自变量栅格路径（键名 = 变量列名）----
x_paths <- c(
  fire_freq     = "data/glm_analysis/fire_frequency.tif",
  drought_freq  = "data/glm_analysis/drought_frequency.tif",
  temp_change   = "data/glm_analysis/global_temperature_trends.tif",
  precip_change = "data/glm_analysis/global_precipitation_trends.tif",
  pop_change    = "data/glm_analysis/population_difference_3km.tif",
  light_change  = "data/glm_analysis/nightlight_change.tif"
)

feat_cols <- names(x_paths)

# 结果表中的变量显示名（与自变量顺序一一对应）
var_names  <- c(
  "Fire frequency", "Drought frequency",
  "Temperature change", "Precipitation change",
  "Population change", "Nighttime light change"
)

# ---- 因变量配置 ----
y_configs <- list(
  list(label = "ALL_COMPOSITE",
       tif = "data/glm_analysis/ALL_composite_diff_all.tif",
       folder = "D:/hzl/code/outputs/GLMM_ALL_COMPOSITE")
)

# ---- 区域配置 ----
region_configs <- list(
  list(label = "AF",     shp = "data/boundaries/AF_final.shp"),
  list(label = "AU",     shp = "data/boundaries/AU.shp"),
  list(label = "China",  shp = "data/boundaries/China.shp"),
  list(label = "EU",     shp = "data/boundaries/EU_final.shp"),
  list(label = "NA",     shp = "data/boundaries/NA_final2.shp"),
  list(label = "RUS",    shp = "data/boundaries/RUS.shp"),
  list(label = "SA",     shp = "data/boundaries/SA.shp"),
  list(label = "SEA",    shp = "data/boundaries/SEA.shp"),
  list(label = "Global", shp = "data/boundaries/countries_boundaries.shp")
)

# ==============================================================================
# 2. 一次性读取所有自变量栅格（固定，不随循环变化）
# ==============================================================================

cat("读取自变量栅格...\n")
x_stack        <- rast(unname(x_paths))
names(x_stack) <- feat_cols
cat("自变量栅格尺寸:", nrow(x_stack), "×", ncol(x_stack), "\n\n")

# ==============================================================================
# 3. 工具函数
# ==============================================================================

# 由经纬度生成 block_id
make_block_id <- function(lon, lat, deg = BLOCK_DEG) {
  blon <- floor(lon / deg) * deg
  blat <- floor(lat / deg) * deg
  paste(blon, blat, sep = "_")
}

# ==============================================================================
# 4. 主循环：因变量 × 区域
# ==============================================================================

for (y_cfg in y_configs) {
  y_label <- y_cfg$label
  out_dir  <- y_cfg$folder
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

  txt_path  <- file.path(out_dir, paste0("GLMM_", y_label, "_results.txt"))
  txt_lines <- character(0)

  cat(strrep("=", 60), "\n")
  cat("因变量:", y_label, "\n")
  cat(strrep("=", 60), "\n")

  y_rast <- rast(y_cfg$tif)

  for (reg_cfg in region_configs) {
    reg_label <- reg_cfg$label
    cat("\n---", y_label, "|", reg_label, "---\n")

    # ---- 读取区域 shapefile ----
    shp <- tryCatch(vect(reg_cfg$shp), error = function(e) {
      cat("  ⚠ shapefile 读取失败:", conditionMessage(e), "\n"); NULL
    })
    if (is.null(shp)) next
    if (!same.crs(shp, x_stack)) shp <- project(shp, crs(x_stack))

    # ---- 裁剪 + 掩膜 X ----
    x_crop <- tryCatch(mask(crop(x_stack, shp), shp), error = function(e) {
      cat("  ⚠ X 裁剪失败:", conditionMessage(e), "\n"); NULL
    })
    if (is.null(x_crop)) next

    # 诊断：X 各层非 NA 像元数
    x_nonNA <- as.integer(global(x_crop[[1]], "notNA")[[1]])
    cat("  [诊断] x_crop 尺寸:", nrow(x_crop), "×", ncol(x_crop),
        " | x[[1]] 非NA:", x_nonNA, "\n")

    # ---- 裁剪 + 掩膜 Y，强制对齐到 X 的网格 ----
    # 将裁剪范围向外扩展两个像元，对负坐标同样适用
    e   <- ext(x_crop)
    buf <- max(res(x_crop)) * 2
    e_expanded <- ext(e$xmin - buf, e$xmax + buf, e$ymin - buf, e$ymax + buf)

    y_crop <- tryCatch({
      y_tmp <- resample(crop(y_rast, e_expanded), x_crop, method = "bilinear")
      mask(y_tmp, shp)
    }, error = function(e_err) {
      cat("  ⚠ Y 裁剪失败:", conditionMessage(e_err), "\n"); NULL
    })
    if (is.null(y_crop)) next

    # 诊断：Y 非 NA 像元数
    y_nonNA <- as.integer(global(y_crop, "notNA")[[1]])
    cat("  [诊断] y_crop 尺寸:", nrow(y_crop), "×", ncol(y_crop),
        " | y 非NA:", y_nonNA, "\n")

    # ---- stack 后一次性转 data.frame，保证 X/Y 像元完全对齐 ----
    combined        <- c(x_crop, y_crop)
    names(combined) <- c(feat_cols, "fragmentation")
    df_raw <- as.data.frame(combined, xy = TRUE, na.rm = FALSE)
    cat("  [诊断] df_raw 行数:", nrow(df_raw),
        " | 完整行数:", sum(complete.cases(df_raw)), "\n")
    df <- na.omit(df_raw)

    if (nrow(df) == 0) {
      na_counts <- colSums(is.na(df_raw))
      cat("  [诊断] 各列 NA 数:\n")
      print(na_counts[na_counts > 0])
      cat("  ⚠ 跳过：无有效像元\n"); next
    }

    # ---- 先抽样，再计算 block_id（减少坐标变换开销）----
    if (is.finite(MAX_OBS) && nrow(df) > MAX_OBS) {
      set.seed(42)
      df <- df[sample(nrow(df), MAX_OBS), ]
    }

    # ---- 生成 5°×5° 空间 block_id（自动处理投影/地理坐标）----
    # 如果栅格是投影坐标系（单位：米），先将 xy 转换为经纬度再分块
    if (is.lonlat(x_stack)) {
      df$block_id <- make_block_id(df$x, df$y)
    } else {
      cat("  [诊断] 投影坐标系，转换为经纬度后分块\n")
      lonlat      <- project(as.matrix(df[, c("x", "y")]),
                             from = crs(x_stack), to = "EPSG:4326")
      df$block_id <- make_block_id(lonlat[, 1], lonlat[, 2])
    }

    # 诊断：显示 block_id 示例与 xy 范围
    cat("  [诊断] xy 范围: x=[", round(min(df$x),2), ",", round(max(df$x),2),
        "]  y=[", round(min(df$y),2), ",", round(max(df$y),2), "]\n")
    cat("  [诊断] block_id 示例:", head(unique(df$block_id), 3), "\n")

    # 剔除像元数不足的 block
    block_n    <- table(df$block_id)
    valid_blks <- names(block_n[block_n >= MIN_BLOCK_N])
    df         <- df[df$block_id %in% valid_blks, ]

    n_obs    <- nrow(df)
    n_blocks <- length(unique(df$block_id))
    cat("  有效像元:", n_obs, " | block 数:", n_blocks, "\n")

    if (n_obs < 30 || n_blocks < 3) {
      cat("  ⚠ 跳过：数据量不足\n"); next
    }

    # ---- 使用 R scale() 按样本标准差标准化 ----
    df[feat_cols]      <- lapply(df[feat_cols], function(x) as.numeric(scale(x)))
    df$fragmentation   <- as.numeric(scale(df$fragmentation))

    # ---- 拟合 GLMM ----
    # 固定效应：6个驱动因子
    # 随机效应：(1 | block_id) —— 表达同一空间块共享的变异
    glmm_formula <- as.formula(
      paste("fragmentation ~",
            paste(feat_cols, collapse = " + "),
            "+ (1 | block_id)")
    )

    cat("  拟合 GLMM...\n")
    model <- tryCatch(
      glmmTMB(glmm_formula, data = df, family = gaussian(),
              control = glmmTMBControl(
                optimizer = optim,
                optArgs   = list(method = "L-BFGS-B")
              )),
      error = function(e) {
        cat("  ⚠ 模型拟合失败:", conditionMessage(e), "\n"); NULL
      }
    )
    if (is.null(model)) next

    # ---- 提取固定效应系数 ----
    fe <- summary(model)$coefficients$cond
    fe <- fe[rownames(fe) != "(Intercept)", , drop = FALSE]

    coef_df <- data.frame(
      Variable    = var_names,
      Coefficient = fe[, "Estimate"],
      Std_Error   = fe[, "Std. Error"],
      p_value     = fe[, "Pr(>|z|)"],
      CI_lower    = fe[, "Estimate"] - 1.96 * fe[, "Std. Error"],
      CI_upper    = fe[, "Estimate"] + 1.96 * fe[, "Std. Error"],
      row.names   = NULL
    )

    print(format(coef_df, digits = 4))

    # ---- 追加到 txt ----
    txt_lines <- c(
      txt_lines,
      strrep("=", 60),
      paste0("区域: ", reg_label, "  |  因变量: ", y_label,
             "  |  n=", n_obs, "  blocks=", n_blocks),
      strrep("=", 60),
      capture.output(print(coef_df, row.names = FALSE, digits = 4)),
      ""
    )

  }

  writeLines(txt_lines, txt_path)
  cat("\nTXT 已保存:", txt_path, "\n")
}

cat("\n全部完成！\n")
cat("输出目录：\n")
for (y_cfg in y_configs) cat(" ", y_cfg$folder, "\n")
