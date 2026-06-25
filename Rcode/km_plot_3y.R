# 主要任务 KM曲线
rm(list = ls())
library(rmda)
library(data.table)
library(Hmisc)
library(rms)
library(readxl)
library(dcurves)
library(caret)
library(survivalROC)
library(timeROC)
library(pROC)
library(glmnet)
library(survival)
library(ROSE)
library(dplyr)
library(survival)
library(survminer)
library(ggplot2)

basereulst <- read_excel('D:/project/0_Competition/data/new_data/DL_RF.xlsx')
save_dir <- 'D:/project/0_Competition/data/new_data/fig/3_y'

train_idx <- grepl('CHUS', basereulst$ID) | grepl('HGJ', basereulst$ID)
inter_idx <- grepl('CHUM', basereulst$ID)
exter_idx <- grepl('HMR', basereulst$ID)

train_data <- basereulst[train_idx, ]
#########
roc_3y <- survivalROC(Stime = train_data$LR_Day, status = train_data$LR,
                      marker = train_data$LR_risk, predict.time = 3 * 365,
                      method = "KM")
youden_index <- roc_3y$TP -  roc_3y$FP
max_idx <- which.max(youden_index)
lrcut_off <- roc_3y$cut.values[max_idx]
LR_data <- data.frame(dataset=basereulst$dataset, Group=basereulst$Group, 
                      ID=basereulst$ID, Time=basereulst$LR_Day, label=basereulst$LR, 
                      sp=ifelse(basereulst$LR_risk >= lrcut_off,'high','low'))

roc_3y <- survivalROC(Stime = train_data$DM_Day, status = train_data$DM,
                      marker = train_data$DM_risk, predict.time = 3 * 365,
                      method = "KM")
youden_index <- roc_3y$TP -  roc_3y$FP
max_idx <- which.max(youden_index)
dmcut_off <- roc_3y$cut.values[max_idx]

DM_data <- data.frame(dataset=basereulst$dataset, Group=basereulst$Group, 
                      ID=basereulst$ID, Time=basereulst$DM_Day, label=basereulst$DM, 
                      sp=ifelse(basereulst$DM_risk >= dmcut_off,'high','low'))


########
## LR train
fit_data <- survfit(Surv(Time, label) ~ sp, data = LR_data[train_idx,])
p <- ggsurvplot(fit_data,
                legend.title = '',
                legend.labs = c("High", "Low"),
                # surv.median.line = 'hv',
                pval=TRUE, # long-rank检验
                xlab = "Time(Days)",
                xlim = c(0, 365*5),
                ylab = "DM-free survival probability",
                risk.table = TRUE, #显示风险表
                risk.table.col = "strata",
                risk.table.y.text.col = TRUE,
                risk.table.fontsize = 6,
                # conf.int = TRUE, # 显示95%置信区间
                surv.scale="percent",
                break.x.by = 365,
                palette="lancet", # 调色板
                size = 2, # 线粗
                censor.size = 6.5 # 图例大小
)

tsize <- 19
tisize <- 16
p$plot <- p$plot + theme(
  legend.text = element_text(size = tisize),
  axis.text.x = element_text(size = tisize, color = "black"),
  axis.text.y = element_text(size = tisize, color = "black"),
  axis.title.x = element_text(size = tsize, color = "black"),
  axis.title.y = element_text(size = tsize, color = "black")
)
p$table <- p$table + theme(
  legend.text = element_text(size = tisize),
  axis.text.x = element_text(size = tisize, color = "black"),
  axis.title.x = element_text(size = tsize, color = "black"),
  axis.title.y = element_text(size = tsize, color = "black")
)
png(file.path(save_dir, 'KM_LR_train.png'), 
    width=700 * 3, height=650 * 3, res = 300, units = 'px')
print(p)
dev.off()

## LR inter
fit_data <- survfit(Surv(Time, label) ~ sp, data = LR_data[inter_idx,])
p <- ggsurvplot(fit_data,
                legend.title = '',
                legend.labs = c("High", "Low"),
                # surv.median.line = 'hv',
                pval=TRUE, # long-rank检验
                xlab = "Time(Days)",
                xlim = c(0, 365*5),
                ylab = "DM-free survival probability",
                risk.table = TRUE, #显示风险表
                risk.table.col = "strata",
                risk.table.y.text.col = TRUE,
                risk.table.fontsize = 6,
                # conf.int = TRUE, # 显示95%置信区间
                surv.scale="percent",
                break.x.by = 365,
                palette="lancet", # 调色板
                size = 2, # 线粗
                censor.size = 6.5 # 图例大小
)

tsize <- 19
tisize <- 16
p$plot <- p$plot + theme(
  legend.text = element_text(size = tisize),
  axis.text.x = element_text(size = tisize, color = "black"),
  axis.text.y = element_text(size = tisize, color = "black"),
  axis.title.x = element_text(size = tsize, color = "black"),
  axis.title.y = element_text(size = tsize, color = "black")
)
p$table <- p$table + theme(
  legend.text = element_text(size = tisize),
  axis.text.x = element_text(size = tisize, color = "black"),
  axis.title.x = element_text(size = tsize, color = "black"),
  axis.title.y = element_text(size = tsize, color = "black")
)
png(file.path(save_dir, 'KM_LR_inter.png'), 
    width=700 * 3, height=650 * 3, res = 300, units = 'px')
print(p)
dev.off()


## LR exter
fit_data <- survfit(Surv(Time, label) ~ sp, data = LR_data[exter_idx,])
p <- ggsurvplot(fit_data,
                legend.title = '',
                legend.labs = c("High", "Low"),
                # surv.median.line = 'hv',
                pval=TRUE, # long-rank检验
                xlab = "Time(Days)",
                xlim = c(0, 365*5),
                ylab = "DM-free survival probability",
                risk.table = TRUE, #显示风险表
                risk.table.col = "strata",
                risk.table.y.text.col = TRUE,
                risk.table.fontsize = 6,
                # conf.int = TRUE, # 显示95%置信区间
                surv.scale="percent",
                break.x.by = 365,
                palette="lancet", # 调色板
                size = 2, # 线粗
                censor.size = 6.5 # 图例大小
)

tsize <- 19
tisize <- 16
p$plot <- p$plot + theme(
  legend.text = element_text(size = tisize),
  axis.text.x = element_text(size = tisize, color = "black"),
  axis.text.y = element_text(size = tisize, color = "black"),
  axis.title.x = element_text(size = tsize, color = "black"),
  axis.title.y = element_text(size = tsize, color = "black")
)
p$table <- p$table + theme(
  legend.text = element_text(size = tisize),
  axis.text.x = element_text(size = tisize, color = "black"),
  axis.title.x = element_text(size = tsize, color = "black"),
  axis.title.y = element_text(size = tsize, color = "black")
)
png(file.path(save_dir, 'KM_LR_exter.png'), 
    width=700 * 3, height=650 * 3, res = 300, units = 'px')
print(p)
dev.off()

########################################################
## DM train
fit_data <- survfit(Surv(Time, label) ~ sp, data = DM_data[train_idx,])
p <- ggsurvplot(fit_data,
                legend.title = '',
                legend.labs = c("High", "Low"),
                # surv.median.line = 'hv',
                pval=TRUE, # long-rank检验
                xlab = "Time(Days)",
                xlim = c(0, 365*5),
                ylab = "DM-free survival probability",
                risk.table = TRUE, #显示风险表
                risk.table.col = "strata",
                risk.table.y.text.col = TRUE,
                risk.table.fontsize = 6,
                # conf.int = TRUE, # 显示95%置信区间
                surv.scale="percent",
                break.x.by = 365,
                palette="lancet", # 调色板
                size = 2, # 线粗
                censor.size = 6.5 # 图例大小
)

tsize <- 19
tisize <- 16
p$plot <- p$plot + theme(
  legend.text = element_text(size = tisize),
  axis.text.x = element_text(size = tisize, color = "black"),
  axis.text.y = element_text(size = tisize, color = "black"),
  axis.title.x = element_text(size = tsize, color = "black"),
  axis.title.y = element_text(size = tsize, color = "black")
)
p$table <- p$table + theme(
  legend.text = element_text(size = tisize),
  axis.text.x = element_text(size = tisize, color = "black"),
  axis.title.x = element_text(size = tsize, color = "black"),
  axis.title.y = element_text(size = tsize, color = "black")
)
png(file.path(save_dir, 'KM_DM_train.png'), 
    width=700 * 3, height=650 * 3, res = 300, units = 'px')
print(p)
dev.off()


## DM inter
fit_data <- survfit(Surv(Time, label) ~ sp, data = DM_data[inter_idx,])
p <- ggsurvplot(fit_data,
                legend.title = '',
                legend.labs = c("High", "Low"),
                # surv.median.line = 'hv',
                pval=TRUE, # long-rank检验
                xlab = "Time(Days)",
                xlim = c(0, 365*5),
                ylab = "DM-free survival probability",
                risk.table = TRUE, #显示风险表
                risk.table.col = "strata",
                risk.table.y.text.col = TRUE,
                risk.table.fontsize = 6,
                # conf.int = TRUE, # 显示95%置信区间
                surv.scale="percent",
                break.x.by = 365,
                palette="lancet", # 调色板
                size = 2, # 线粗
                censor.size = 6.5 # 图例大小
)

tsize <- 19
tisize <- 16
p$plot <- p$plot + theme(
  legend.text = element_text(size = tisize),
  axis.text.x = element_text(size = tisize, color = "black"),
  axis.text.y = element_text(size = tisize, color = "black"),
  axis.title.x = element_text(size = tsize, color = "black"),
  axis.title.y = element_text(size = tsize, color = "black")
)
p$table <- p$table + theme(
  legend.text = element_text(size = tisize),
  axis.text.x = element_text(size = tisize, color = "black"),
  axis.title.x = element_text(size = tsize, color = "black"),
  axis.title.y = element_text(size = tsize, color = "black")
)
png(file.path(save_dir, 'KM_DM_inter.png'), 
    width=700 * 3, height=650 * 3, res = 300, units = 'px')
print(p)
dev.off()


## DM exter
fit_data <- survfit(Surv(Time, label) ~ sp, data = DM_data[exter_idx,])
p <- ggsurvplot(fit_data,
                legend.title = '',
                legend.labs = c("High", "Low"),
                # surv.median.line = 'hv',
                pval=TRUE, # long-rank检验
                xlab = "Time(Days)",
                xlim = c(0, 365*5),
                ylab = "DM-free survival probability",
                risk.table = TRUE, #显示风险表
                risk.table.col = "strata",
                risk.table.y.text.col = TRUE,
                risk.table.fontsize = 6,
                # conf.int = TRUE, # 显示95%置信区间
                surv.scale="percent",
                break.x.by = 365,
                palette="lancet", # 调色板
                size = 2, # 线粗
                censor.size = 6.5 # 图例大小
)

tsize <- 19
tisize <- 16
p$plot <- p$plot + theme(
  legend.text = element_text(size = tisize),
  axis.text.x = element_text(size = tisize, color = "black"),
  axis.text.y = element_text(size = tisize, color = "black"),
  axis.title.x = element_text(size = tsize, color = "black"),
  axis.title.y = element_text(size = tsize, color = "black")
)
p$table <- p$table + theme(
  legend.text = element_text(size = tisize),
  axis.text.x = element_text(size = tisize, color = "black"),
  axis.title.x = element_text(size = tsize, color = "black"),
  axis.title.y = element_text(size = tsize, color = "black")
)
png(file.path(save_dir, 'KM_DM_exter.png'), 
    width=700 * 3, height=650 * 3, res = 300, units = 'px')
print(p)
dev.off()
