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

basereulst <- read_excel('D:/project/0_Competition/data/new_data/DL_RF.xlsx')
save_dir <- 'D:/project/0_Competition/data/new_data/output'

train_idx <- grepl('CHUS', basereulst$ID) | grepl('HGJ', basereulst$ID)
inter_idx <- grepl('CHUM', basereulst$ID)
exter_idx <- grepl('HMR', basereulst$ID)

#####################################
# DCA
#####################################
library(riskRegression)
library(ResourceSelection)
library(rms)
library(gridExtra)
library(survival)

plot_data = basereulst[exter_idx,]
dd <- datadist(plot_data)
options(datadist = 'dd')


library(rmda)
fit_data <- function(fit1, fit2, name1, name2){
  fit1_list = split(fit1$derived.data, fit1$derived.data$model)
  fit2_list = split(fit2$derived.data, fit2$derived.data$model)
  
  basedata <- data.frame(
    thresholds=fit1_list[[1]]$thresholds,
    alld=fit1_list[['All']]$sNB,
    Noned=fit1_list[['None']]$sNB,
    model1=fit1_list[[name1]]$sNB,
    model2=fit2_list[[name2]]$sNB
  )
  
  colnames(basedata) <- c('thresholds', 'All', 'None', name1, name2)
  return (basedata)
}

plot_data$DM <- ifelse(plot_data$DM == 0,0,1)
plot_data$LR <- ifelse(plot_data$LR == 0,0,1)

########## DM
time_points = 365 * 3

fit_DL <- cph(Surv(DM_Day, DM) ~ DM_risk, data = plot_data, x=TRUE, y=TRUE, surv=TRUE)
plot_data$DM_probDL <- 1 - survfit(fit_DL, newdata = plot_data, se.fit = FALSE)$surv[which.min(abs(survfit(fit_DL)$time - time_points)), ]
fit_RF <- cph(Surv(DM_Day, DM) ~ DM_rf, data = plot_data, x=TRUE, y=TRUE, surv=TRUE)
plot_data$DM_probRF <- 1 - survfit(fit_RF, newdata = plot_data, se.fit = FALSE)$surv[which.min(abs(survfit(fit_RF)$time - time_points)), ]

fit1 <- decision_curve(DM ~ DM_probDL, data = plot_data,
                       study.design = 'cohort', bootstraps = 100)
fit2 <- decision_curve(DM ~ DM_probRF, data = plot_data,
                       study.design = 'cohort', bootstraps = 100)
dmdata <- fit_data(fit1, fit2, 'DM ~ DM_probDL', 'DM ~ DM_probRF')
na.omit(dmdata)
colnames(dmdata)[4] = 'DL'
colnames(dmdata)[5] = 'RF'

pd <- ggplot(data = dmdata, aes(x = thresholds)) +
  geom_line(aes(y = All, color = 'All'), lwd = 1.5, linetype = 1) +
  geom_line(aes(y = None, color = 'None'), lwd = 1.5, linetype = 1) +
  geom_smooth(aes(y = DL, color = 'Deep learning'), span=0.5, se=FALSE, 
              lwd = 1.5, linetype = 1) +
  geom_smooth(aes(y = RF, color = 'Radiomics'), span=0.5, se=FALSE,
              lwd = 1.5, linetype = 1) +
  theme_bw() + labs(x = "Threshold probability", y = "Net benefit") +
  scale_color_manual('', values = c('All'='black','None' = 'gray',
                                    'Deep learning' = "#BC3C29FF", 
                                    'Radiomics' = "#0072B5FF"),
                     breaks = c('Deep learning', 'Radiomics','All','None'))+
  scale_x_continuous(limits = c(0, 1.0), breaks = seq(0,1.0, 0.2), minor_breaks = NULL) + 
  scale_y_continuous(limits = c(-0.2, 1.0), breaks = seq(-0.2, 1.0, 0.2), minor_breaks = NULL)+
  theme(
    axis.text = element_text(size = 15, color = "black"),
    axis.title.x = element_text(size = 14, color = "black", margin = margin(c(15, 0, 0, 0))),
    axis.title.y = element_text(size = 14, color = "black", margin = margin(c(0, 15, 0, 0))),
    legend.title = element_blank(), 
    legend.position = c(0.85,0.85),legend.background = element_rect(colour = 'black'),
    legend.text = element_text(size = 12), legend.key.width = unit(10, 'mm'),
    panel.border = element_blank(), axis.line = element_line(colour = "black", linewidth = 1)
  )
png(file.path(save_dir, 'DCA_DM.png'), width=720 * 3, height=640 * 3, res = 300, units = 'px')
print(pd)
dev.off()

########## LR
# 将risk转换为01概率
fit_DL <- cph(Surv(LR_Day, LR) ~ LR_risk, data = plot_data, x=TRUE, y=TRUE, surv=TRUE)
plot_data$LR_probDL <- 1 - survfit(fit_DL, newdata = plot_data, se.fit = FALSE)$surv[which.min(abs(survfit(fit_DL)$time - time_points)), ]
fit_RF <- cph(Surv(LR_Day, LR) ~ LR_rf, data = plot_data, x=TRUE, y=TRUE, surv=TRUE)
plot_data$LR_probRF <- 1 - survfit(fit_RF, newdata = plot_data, se.fit = FALSE)$surv[which.min(abs(survfit(fit_RF)$time - time_points)), ]

fit1 <- decision_curve(LR ~ LR_probDL, data = plot_data,
                       study.design = 'cohort', bootstraps = 100)
fit2 <- decision_curve(LR ~ LR_probRF, data = plot_data,
                       study.design = 'cohort', bootstraps = 100)
dmdata <- fit_data(fit1, fit2, 'LR ~ LR_probDL', 'LR ~ LR_probRF')
na.omit(dmdata)
colnames(dmdata)[4] = 'DL'
colnames(dmdata)[5] = 'RF'

pd <- ggplot(data = dmdata, aes(x = thresholds)) +
  geom_line(aes(y = All, color = 'All'), lwd = 1.5, linetype = 1) +
  geom_line(aes(y = None, color = 'None'), lwd = 1.5, linetype = 1) +
  geom_smooth(aes(y = DL, color = 'Deep learning'), span=0.5, se=FALSE, 
              lwd = 1.5, linetype = 1) +
  geom_smooth(aes(y = RF, color = 'Radiomics'), span=0.5, se=FALSE,
              lwd = 1.5, linetype = 1) +
  theme_bw() + labs(x = "Threshold probability", y = "Net benefit") +
  scale_color_manual('', values = c('All'='black','None' = 'gray',
                                    'Deep learning' = "#BC3C29FF", 
                                    'Radiomics' = "#0072B5FF"),
                     breaks = c('Deep learning', 'Radiomics','All','None'))+
  scale_x_continuous(limits = c(0, 1.0), breaks = seq(0,1.0, 0.2), minor_breaks = NULL) + 
  scale_y_continuous(limits = c(-0.2, 1.0), breaks = seq(-0.2, 1.0, 0.2), minor_breaks = NULL)+
  theme(
    axis.text = element_text(size = 15, color = "black"),
    axis.title.x = element_text(size = 14, color = "black", margin = margin(c(15, 0, 0, 0))),
    axis.title.y = element_text(size = 14, color = "black", margin = margin(c(0, 15, 0, 0))),
    legend.title = element_blank(), 
    legend.position = c(0.85,0.85),legend.background = element_rect(colour = 'black'),
    legend.text = element_text(size = 12), legend.key.width = unit(10, 'mm'),
    panel.border = element_blank(), axis.line = element_line(colour = "black", linewidth = 1)
  )
png(file.path(save_dir, 'DCA_LR.png'), width=720 * 3, height=640 * 3, res = 300, units = 'px')
print(pd)
dev.off()

#####################################
# 柱状图
#####################################
library(ggprism)
get_sen_setspe <- function(Sen, Spe, setcut){
  sen_results <- vector(mode="numeric",length=0)
  for (t in 1:ncol(Spe)){
    Sen_index <-  which.min(abs(Spe[,t] - setcut))
    sen_results[t] <- Sen[Sen_index, t]
  }
  return(sen_results)
}

get_spe_setsen <- function(Sen, Spe, setcut){
  spe_results <- vector(mode="numeric",length=0)
  for (t in 1:ncol(Sen)){
    Spe_index <-  which.min(abs(Sen[,t] - setcut))
    spe_results[t] <- Spe[Spe_index, t]
  }
  return(spe_results)
}

time_points = c(2,3)*365

########### DM
ci_DM_risk <- 1-concordance(Surv(DM_Day, DM) ~ DM_risk, data = plot_data)$concordance
TROC_DM_risk <- timeROC(T=plot_data$DM_Day,delta=plot_data$DM,marker=plot_data$DM_risk,
                        cause=1,weighting="marginal",times=time_points,ROC = TRUE,iid = FALSE)
Tspe_DM_risk <- get_spe_setsen(TROC_DM_risk$TP, 1 - TROC_DM_risk$FP, 0.8)

ci_DM_rf <- 1-concordance(Surv(DM_Day, DM) ~ DM_rf, data = plot_data)$concordance
TROC_DM_rf <- timeROC(T=plot_data$DM_Day,delta=plot_data$DM,marker=plot_data$DM_rf,
                      cause=1,weighting="marginal",times=time_points,ROC = TRUE,iid = FALSE)
Tspe_DM_rf <- get_spe_setsen(TROC_DM_rf$TP, 1 - TROC_DM_rf$FP, 0.8)

metric_DM <- data.frame(
  item=rep(c('C-index', 'TAUC_2y', 'TAUC_3y', 'TSpe_2y', 'TSpe_3y'), 2),
  group=c(rep(c('Deep Learning'), 5), rep(c('Radimocs'), 5)),
  value=c(ci_DM_risk, TROC_DM_risk$AUC, Tspe_DM_risk, ci_DM_rf, TROC_DM_rf$AUC, Tspe_DM_rf)
)

metric_DM$item <- factor(metric_DM$item, levels=c('C-index','TAUC_2y','TAUC_3y','TSpe_2y', 'TSpe_3y'))
gp <- ggplot(metric_DM, aes(x=item, y=value, fill=group))+
  geom_bar(stat='identity', position='dodge', width=0.8)+
  theme_prism(base_fontface = "plain", # 字体样式，可选 bold, plain, italic
              # base_family = "serif", # 字体格式，可选 serif, sans, mono, Arial等
              base_size = 15,  # 图形的字体大小
              base_line_size = 0.8, # 坐标轴的粗细
              axis_text_angle = 0) +
  scale_fill_manual(values=rep(c('#F6D1D8', '#CDE7F8'), 5))+
  # scale_color_manual(values=rep(c('#BC3C29FF', '#0072B5FF'), 5))+
  theme(legend.position = "top", legend.text = element_text(size = 15))+
  xlab(NULL)+ylab(NULL)+
  scale_y_continuous(limits = c(0, 1), breaks = seq(0,1,0.2))
png(file.path(save_dir, 'metric_DM.png'), width=760 * 3, height=640 * 3, res = 300, units = 'px')
print(gp)
dev.off()

########### LR
ci_LR_risk <- 1-concordance(Surv(LR_Day, LR) ~ LR_risk, data = plot_data)$concordance
TROC_LR_risk <- timeROC(T=plot_data$LR_Day,delta=plot_data$LR,marker=plot_data$LR_risk,
                        cause=1,weighting="marginal",times=time_points,ROC = TRUE,iid = FALSE)
Tspe_LR_risk <- get_spe_setsen(TROC_LR_risk$TP, 1 - TROC_LR_risk$FP, 0.8)

ci_LR_rf <- 1-concordance(Surv(LR_Day, LR) ~ LR_rf, data = plot_data)$concordance
TROC_LR_rf <- timeROC(T=plot_data$LR_Day,delta=plot_data$LR,marker=plot_data$LR_rf,
                      cause=1,weighting="marginal",times=time_points,ROC = TRUE,iid = FALSE)
Tspe_LR_rf <- get_spe_setsen(TROC_LR_rf$TP, 1 - TROC_LR_rf$FP, 0.8)

metric_LR <- data.frame(
  item=rep(c('C-index', 'TAUC_2y', 'TAUC_3y', 'TSpe_2y', 'TSpe_3y'), 2),
  group=c(rep(c('Deep Learning'), 5), rep(c('Radimocs'), 5)),
  value=c(ci_LR_risk, TROC_LR_risk$AUC, Tspe_LR_risk, ci_LR_rf, TROC_LR_rf$AUC, Tspe_LR_rf)
)

metric_LR$item <- factor(metric_LR$item, levels=c('C-index','TAUC_2y','TAUC_3y', 'TSpe_2y', 'TSpe_3y'))
gp <- ggplot(metric_LR, aes(x=item, y=value, fill=group))+
  geom_bar(stat='identity', position='dodge', width=0.8)+
  theme_prism(base_fontface = "plain", # 字体样式，可选 bold, plain, italic
              # base_family = "serif", # 字体格式，可选 serif, sans, mono, Arial等
              base_size = 15,  # 图形的字体大小
              base_line_size = 0.8, # 坐标轴的粗细
              axis_text_angle = 0) +
  scale_fill_manual(values=rep(c('#F6D1D8', '#CDE7F8'), 5))+
  # scale_color_manual(values=rep(c('#BC3C29FF', '#0072B5FF'), 5))+
  theme(legend.position = "top", legend.text = element_text(size = 15))+
  xlab(NULL)+ylab(NULL)+
  scale_y_continuous(limits = c(0, 1), breaks = seq(0,1,0.2))
png(file.path(save_dir, 'metric_LR.png'), width=760 * 3, height=640 * 3, res = 300, units = 'px')
print(gp)
dev.off()

#####################################
# TimeROC
#####################################
timeroc_DM <- data.frame(
  TP_2year_D = TROC_DM_risk$TP[, 1],
  FP_2year_D = TROC_DM_risk$FP[, 1],
  TP_3year_D = TROC_DM_risk$TP[, 2],
  FP_3year_D = TROC_DM_risk$FP[, 2],
  TP_2year_R = TROC_DM_rf$TP[, 1],
  FP_2year_R = TROC_DM_rf$FP[, 1],
  TP_3year_R = TROC_DM_rf$TP[, 2],
  FP_3year_R = TROC_DM_rf$FP[, 2]
)

library(ggplot2)
gp <- ggplot(data = timeroc_DM) +
  geom_line(aes(x = FP_2year_D, y = TP_2year_D), size = 1.5, color = "#BC3C29FF") +
  geom_line(aes(x = FP_3year_D, y = TP_3year_D), size = 1.5, color = "#BC3C29FF", linetype = 2) +
  geom_line(aes(x = FP_2year_R, y = TP_2year_R), size = 1.5, color = "#0072B5FF") +
  geom_line(aes(x = FP_3year_R, y = TP_3year_R), size = 1.5, color = "#0072B5FF", linetype = 2) +
  geom_abline(slope = 1, intercept = 0, color = "grey", size = 1, linetype = 2) +
  theme_bw() +
  annotate("text",
           x = 0.6, y = 0.15, size = 4.5,
           label = paste0("AUC at 2 years: DL model=", sprintf("%.2f", TROC_DM_risk$AUC[[1]]),
                          ", R model=", sprintf("%.2f", TROC_DM_rf$AUC[[1]]))
  ) +
  annotate("text",
           x = 0.6, y = 0.10, size = 4.5,
           label = paste0("AUC at 3 years: DL model=", sprintf("%.2f", TROC_DM_risk$AUC[[2]]),
                          ", R model=", sprintf("%.2f", TROC_DM_rf$AUC[[2]]))
  ) +
  labs(x = "False positive rate", y = "True positive rate") +
  theme(
    axis.text = element_text(size = 11, color = "black"),
    axis.title.x = element_text(size = 14, color = "black", margin = margin(c(15, 0, 0, 0))),
    axis.title.y = element_text(size = 14, color = "black", margin = margin(c(0, 15, 0, 0))),
    panel.grid = element_blank(),
  ) +
  scale_x_continuous(expand = c(0.01, 0.01), limits = c(0, 1)) + 
  scale_y_continuous(expand = c(0.01, 0.01), limits = c(0, 1))

png(file.path(save_dir, 'TimeROC_DM.png'), width=700 * 3, height=640 * 3, res = 300, units = 'px')
print(gp)
dev.off()

############ LR
timeroc_LR <- data.frame(
  TP_2year_D = TROC_LR_risk$TP[, 1],
  FP_2year_D = TROC_LR_risk$FP[, 1],
  TP_3year_D = TROC_LR_risk$TP[, 2],
  FP_3year_D = TROC_LR_risk$FP[, 2],
  TP_2year_R = TROC_LR_rf$TP[, 1],
  FP_2year_R = TROC_LR_rf$FP[, 1],
  TP_3year_R = TROC_LR_rf$TP[, 2],
  FP_3year_R = TROC_LR_rf$FP[, 2]
)

library(ggplot2)
gp <- ggplot(data = timeroc_LR) +
  geom_line(aes(x = FP_2year_D, y = TP_2year_D), size = 1.5, color = "#BC3C29FF") +
  geom_line(aes(x = FP_3year_D, y = TP_3year_D), size = 1.5, color = "#BC3C29FF", linetype = 2) +
  geom_line(aes(x = FP_2year_R, y = TP_2year_R), size = 1.5, color = "#0072B5FF") +
  geom_line(aes(x = FP_3year_R, y = TP_3year_R), size = 1.5, color = "#0072B5FF", linetype = 2) +
  geom_abline(slope = 1, intercept = 0, color = "grey", size = 1, linetype = 2) +
  theme_bw() +
  annotate("text",
           x = 0.6, y = 0.15, size = 4.5,
           label = paste0("AUC at 2 years: DL model=", sprintf("%.2f", TROC_LR_risk$AUC[[1]]),
                          ", R model=", sprintf("%.2f", TROC_LR_rf$AUC[[1]]))
  ) +
  annotate("text",
           x = 0.6, y = 0.10, size = 4.5,
           label = paste0("AUC at 3 years: DL model=", sprintf("%.2f", TROC_LR_risk$AUC[[2]]),
                          ", R model=", sprintf("%.2f", TROC_LR_rf$AUC[[2]]))
  ) +
  labs(x = "False positive rate", y = "True positive rate") +
  theme(
    axis.text = element_text(size = 11, color = "black"),
    axis.title.x = element_text(size = 14, color = "black", margin = margin(c(15, 0, 0, 0))),
    axis.title.y = element_text(size = 14, color = "black", margin = margin(c(0, 15, 0, 0))),
    panel.grid = element_blank(),
  ) +
  scale_x_continuous(expand = c(0.01, 0.01), limits = c(0, 1)) + 
  scale_y_continuous(expand = c(0.01, 0.01), limits = c(0, 1))

png(file.path(save_dir, 'TimeROC_LR.png'), width=700 * 3, height=640 * 3, res = 300, units = 'px')
print(gp)
dev.off()




# 1,3,5
timeroc_DM <- data.frame(
  TP_1year_D = TROC_DM_risk$TP[, 1],
  FP_1year_D = TROC_DM_risk$FP[, 1],
  TP_3year_D = TROC_DM_risk$TP[, 2],
  FP_3year_D = TROC_DM_risk$FP[, 2],
  TP_5year_D = TROC_DM_risk$TP[, 3],
  FP_5year_D = TROC_DM_risk$FP[, 3],
  TP_1year_R = TROC_DM_rf$TP[, 1],
  FP_1year_R = TROC_DM_rf$FP[, 1],
  TP_3year_R = TROC_DM_rf$TP[, 2],
  FP_3year_R = TROC_DM_rf$FP[, 2],
  TP_5year_R = TROC_DM_rf$TP[, 3],
  FP_5year_R = TROC_DM_rf$FP[, 3]
)

library(ggplot2)
gp <- ggplot(data = timeroc_DM) +
  geom_line(aes(x = FP_1year_D, y = TP_1year_D), size = 1.5, color = "#BC3C29FF") +
  geom_line(aes(x = FP_3year_D, y = TP_3year_D), size = 1.5, color = "#BC3C29FF", linetype = 2) +
  geom_line(aes(x = FP_5year_D, y = TP_5year_D), size = 1.5, color = "#BC3C29FF", linetype = 3) +
  geom_line(aes(x = FP_1year_R, y = TP_1year_R), size = 1.5, color = "#0072B5FF") +
  geom_line(aes(x = FP_3year_R, y = TP_3year_R), size = 1.5, color = "#0072B5FF", linetype = 2) +
  geom_line(aes(x = FP_5year_R, y = TP_5year_R), size = 1.5, color = "#0072B5FF", linetype = 3) +
  geom_abline(slope = 1, intercept = 0, color = "grey", size = 1, linetype = 2) +
  theme_bw() +
  annotate("text",
           x = 0.6, y = 0.15, size = 4.5,
           label = paste0("AUC at 1 years: DL model=", sprintf("%.2f", TROC_DM_risk$AUC[[1]]),
                          ", R model=", sprintf("%.2f", TROC_DM_rf$AUC[[1]]))
  ) +
  annotate("text",
           x = 0.6, y = 0.10, size = 4.5,
           label = paste0("AUC at 3 years: DL model=", sprintf("%.2f", TROC_DM_risk$AUC[[2]]),
                          ", R model=", sprintf("%.2f", TROC_DM_rf$AUC[[2]]))
  ) +
  annotate("text",
           x = 0.6, y = 0.05, size = 4.5,
           label = paste0("AUC at 5 years: DL model=", sprintf("%.2f", TROC_DM_risk$AUC[[3]]),
                          ", R model=", sprintf("%.2f", TROC_DM_rf$AUC[[3]]))
  ) +
  labs(x = "False positive rate", y = "True positive rate") +
  theme(
    axis.text = element_text(size = 11, color = "black"),
    axis.title.x = element_text(size = 14, color = "black", margin = margin(c(15, 0, 0, 0))),
    axis.title.y = element_text(size = 14, color = "black", margin = margin(c(0, 15, 0, 0))),
    panel.grid = element_blank(),
  ) +
  scale_x_continuous(expand = c(0.01, 0.01), limits = c(0, 1)) + 
  scale_y_continuous(expand = c(0.01, 0.01), limits = c(0, 1))

png(file.path(save_dir, 'TimeROC_DM.png'), width=700 * 3, height=640 * 3, res = 300, units = 'px')
print(gp)
dev.off()

############ LR
timeroc_LR <- data.frame(
  TP_1year_D = TROC_LR_risk$TP[, 1],
  FP_1year_D = TROC_LR_risk$FP[, 1],
  TP_3year_D = TROC_LR_risk$TP[, 2],
  FP_3year_D = TROC_LR_risk$FP[, 2],
  TP_5year_D = TROC_LR_risk$TP[, 3],
  FP_5year_D = TROC_LR_risk$FP[, 3],
  TP_1year_R = TROC_LR_rf$TP[, 1],
  FP_1year_R = TROC_LR_rf$FP[, 1],
  TP_3year_R = TROC_LR_rf$TP[, 2],
  FP_3year_R = TROC_LR_rf$FP[, 2],
  TP_5year_R = TROC_LR_rf$TP[, 3],
  FP_5year_R = TROC_LR_rf$FP[, 3]
)

library(ggplot2)
gp <- ggplot(data = timeroc_LR) +
  geom_line(aes(x = FP_1year_D, y = TP_1year_D), size = 1.5, color = "#BC3C29FF") +
  geom_line(aes(x = FP_3year_D, y = TP_3year_D), size = 1.5, color = "#BC3C29FF", linetype = 2) +
  geom_line(aes(x = FP_5year_D, y = TP_5year_D), size = 1.5, color = "#BC3C29FF", linetype = 3) +
  geom_line(aes(x = FP_1year_R, y = TP_1year_R), size = 1.5, color = "#0072B5FF") +
  geom_line(aes(x = FP_3year_R, y = TP_3year_R), size = 1.5, color = "#0072B5FF", linetype = 2) +
  geom_line(aes(x = FP_5year_R, y = TP_5year_R), size = 1.5, color = "#0072B5FF", linetype = 3) +
  geom_abline(slope = 1, intercept = 0, color = "grey", size = 1, linetype = 2) +
  theme_bw() +
  annotate("text",
           x = 0.6, y = 0.15, size = 4.5,
           label = paste0("AUC at 1 years: DL model=", sprintf("%.2f", TROC_LR_risk$AUC[[1]]),
                          ", R model=", sprintf("%.2f", TROC_LR_rf$AUC[[1]]))
  ) +
  annotate("text",
           x = 0.6, y = 0.10, size = 4.5,
           label = paste0("AUC at 3 years: DL model=", sprintf("%.2f", TROC_LR_risk$AUC[[2]]),
                          ", R model=", sprintf("%.2f", TROC_LR_rf$AUC[[2]]))
  ) +
  annotate("text",
           x = 0.6, y = 0.05, size = 4.5,
           label = paste0("AUC at 5 years: DL model=", sprintf("%.2f", TROC_LR_risk$AUC[[3]]),
                          ", R model=", sprintf("%.2f", TROC_LR_rf$AUC[[3]]))
  ) +
  labs(x = "False positive rate", y = "True positive rate") +
  theme(
    axis.text = element_text(size = 11, color = "black"),
    axis.title.x = element_text(size = 14, color = "black", margin = margin(c(15, 0, 0, 0))),
    axis.title.y = element_text(size = 14, color = "black", margin = margin(c(0, 15, 0, 0))),
    panel.grid = element_blank(),
  ) +
  scale_x_continuous(expand = c(0.01, 0.01), limits = c(0, 1)) + 
  scale_y_continuous(expand = c(0.01, 0.01), limits = c(0, 1))

png(file.path(save_dir, 'TimeROC_LR.png'), width=700 * 3, height=640 * 3, res = 300, units = 'px')
print(gp)
dev.off()
