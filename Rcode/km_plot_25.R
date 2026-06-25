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
save_dir <- 'D:/project/0_Competition/data/new_data/fig/25cut'

train_idx <- grepl('CHUS', basereulst$ID) | grepl('HGJ', basereulst$ID)
inter_idx <- grepl('CHUM', basereulst$ID)
exter_idx <- grepl('HMR', basereulst$ID)

train_data <- basereulst[train_idx, ]
inter_data <- basereulst[inter_idx, ]
exter_data <- basereulst[exter_idx, ]

CUT_POINT = 0.75

########
## LR train
cut_off <- quantile(train_data$LR_risk, CUT_POINT, na.rm = TRUE)
km_data <- data.frame(Time=train_data$LR_Day, label=train_data$LR, 
                      sp=ifelse(train_data$LR_risk > cut_off,'high','low'))
fit_data <- survfit(Surv(Time, label) ~ sp, data = km_data)
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
cut_off <- quantile(inter_data$LR_risk, CUT_POINT, na.rm = TRUE)
km_data <- data.frame(Time=inter_data$LR_Day, label=inter_data$LR, 
                      sp=ifelse(inter_data$LR_risk > cut_off,'high','low'))
fit_data <- survfit(Surv(Time, label) ~ sp, data = km_data)
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
cut_off <- quantile(exter_data$LR_risk, CUT_POINT, na.rm = TRUE)
km_data <- data.frame(Time=exter_data$LR_Day, label=exter_data$LR, 
                      sp=ifelse(exter_data$LR_risk > cut_off,'high','low'))
fit_data <- survfit(Surv(Time, label) ~ sp, data = km_data)
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
cut_off <- quantile(train_data$DM_risk, CUT_POINT, na.rm = TRUE)
km_data <- data.frame(Time=train_data$DM_Day, label=train_data$DM, 
                      sp=ifelse(train_data$DM_risk > cut_off,'high','low'))
fit_data <- survfit(Surv(Time, label) ~ sp, data = km_data)
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
cut_off <- quantile(inter_data$DM_risk, CUT_POINT, na.rm = TRUE)
km_data <- data.frame(Time=inter_data$DM_Day, label=inter_data$DM, 
                      sp=ifelse(inter_data$DM_risk > cut_off,'high','low'))
fit_data <- survfit(Surv(Time, label) ~ sp, data = km_data)
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
cut_off <- quantile(exter_data$DM_risk, CUT_POINT, na.rm = TRUE)
km_data <- data.frame(Time=exter_data$DM_Day, label=exter_data$DM, 
                      sp=ifelse(exter_data$DM_risk > cut_off,'high','low'))
fit_data <- survfit(Surv(Time, label) ~ sp, data = km_data)
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