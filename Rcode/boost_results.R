rm(list = ls())
library(rmda)
library(data.table)
library(readxl)
library(survival)
library(timeROC)
library(boot)
library(pROC)

library(Hmisc)
library(rms)
library(dcurves)
library(caret)
library(survivalROC)
library(glmnet)
library(openxlsx)
library(tibble)

pd_data <- read_excel('D:/project/0_Competition/data/new_data/DL_RF.xlsx')
save_dir <- 'D:/project/0_Competition/data/new_data/output'
n_bootstrap = 1000
inputtime = c(2, 3)

boost_evaluation <- function(train_data, event, event_time, marker, n_bootstrap=1000, inputtime=c(1,3,5)){
  time_points = inputtime * 365
  c_index_results <- numeric(n_bootstrap)
  auc_results <- matrix(NA, nrow = n_bootstrap, ncol = length(time_points))
  sen_results <- matrix(NA, nrow = n_bootstrap, ncol = length(time_points))
  spe_results <- matrix(NA, nrow = n_bootstrap, ncol = length(time_points))
  
  surv_for <- as.formula(paste0('Surv(',event_time, ',', event, ') ~ ', marker))
  
  i = 1
  set.seed(123)
  while (i <= n_bootstrap){
    sample_data <- train_data[sample(nrow(train_data), replace = TRUE), ]
    if (length(unique(sample_data[[event]])) == 2){
      TimeROC_results <- timeROC(T=sample_data[[event_time]],
                                 delta=sample_data[[event]],
                                 marker=sample_data[[marker]],
                                 cause=1,weighting="marginal",
                                 times=time_points,ROC = TRUE,iid = FALSE)
      if (!any(is.na(TimeROC_results$AUC))){
        c_index_results[i] <- 1-concordance(surv_for, data = sample_data)$concordance
        auc_results[i,] <- TimeROC_results$AUC
        
        # 计算每个时间点上，SEN = 0.8对应的SPE
        Sen <- TimeROC_results$TP
        Spe <- 1 - TimeROC_results$FP
        
        for (t in 1:length(time_points)){
          Spe_index <- which.min(abs(Sen[,t] - 0.8))
          spe_results[i, t] <- Spe[Spe_index, t]
          
          Sen_index <-  which.min(abs(Spe[,t] - 0.8))
          sen_results[i, t] <- Sen[Sen_index, t]
        }
        i = i + 1
      }
    }
  }
  
  ci_index_ci <- quantile(c_index_results, c(0.025, 0.975))
  time_auc_ci <- apply(auc_results, 2, function(x) quantile(x, c(0.025, 0.975)))
  time_sen_ci <- apply(sen_results, 2, function(x) quantile(x, c(0.025, 0.975)))
  time_spe_ci <- apply(spe_results, 2, function(x) quantile(x, c(0.025, 0.975)))
  cidata <- rbind(ci_index_ci, t(time_auc_ci), t(time_sen_ci), t(time_spe_ci))
  
  ci_index_mean <- mean(c_index_results)
  time_auc_mean <- apply(auc_results, 2, function(x) mean(x))
  time_sen_mean <- apply(sen_results, 2, function(x) mean(x))
  time_spe_mean <- apply(spe_results, 2, function(x) mean(x))
  meandata <- cbind(ci_index_mean, t(time_auc_mean), t(time_sen_mean), t(time_spe_mean)) %>% t()
  
  ci_index_base <- 1-concordance(surv_for, data = train_data)$concordance
  TimeROCbase <- timeROC(T=train_data[[event_time]],
                         delta=train_data[[event]],
                         marker=train_data[[marker]],
                         cause=1,weighting="marginal",
                         times=time_points,ROC = TRUE,iid = FALSE)
  time_auc_base <- TimeROCbase$AUC
  time_sen_base <- matrix(NA, nrow = 1, ncol = length(time_points))
  time_spe_base <- matrix(NA, nrow = 1, ncol = length(time_points))
  
  Senbase <- TimeROCbase$TP
  Spebase <- 1 - TimeROCbase$FP
  for (t in 1:length(time_points)){
    Spe_index <- which.min(abs(Senbase[,t] - 0.8))
    time_spe_base[1, t] <- Spebase[Spe_index, t]
    
    Sen_index <-  which.min(abs(Spebase[,t] - 0.8))
    time_sen_base[1, t] <- Senbase[Sen_index, t]
  }
  basedata <- cbind(ci_index_base, t(time_auc_base), time_sen_base, time_spe_base) %>% t()
  
  allresult <- cbind(basedata, meandata, cidata) %>% as.data.frame()
  colnames(allresult) <-c('base', 'mean', '2.5%', '97.5%')
  rownames(allresult) <- c("Cindex", paste0("AUC_", inputtime), paste0("Sen_", inputtime), paste0("Sep_", inputtime))
  
  df_fixed <- allresult %>% 
    rownames_to_column(var = "Item")
  return(df_fixed)
}

################### deep learning output
train_data <- pd_data[pd_data$dataset == 'Train',]
train_DMrisk <- boost_evaluation(train_data, 'DM', 'DM_Day', 'DM_risk', n_bootstrap = n_bootstrap, inputtime = inputtime)
train_LRrisk <- boost_evaluation(train_data, 'LR', 'LR_Day', 'LR_risk', n_bootstrap = n_bootstrap, inputtime = inputtime)

inter_data <- pd_data[pd_data$dataset == 'Inter',]
inter_DMrisk <- boost_evaluation(inter_data, 'DM', 'DM_Day', 'DM_risk', n_bootstrap = n_bootstrap, inputtime = inputtime)
inter_LRrisk <- boost_evaluation(inter_data, 'LR', 'LR_Day', 'LR_risk', n_bootstrap = n_bootstrap, inputtime = inputtime)

exter_data <- pd_data[pd_data$dataset == 'Exter',]
exter_DMrisk <- boost_evaluation(exter_data, 'DM', 'DM_Day', 'DM_risk', n_bootstrap = n_bootstrap, inputtime = inputtime)
exter_LRrisk <- boost_evaluation(exter_data, 'LR', 'LR_Day', 'LR_risk', n_bootstrap = n_bootstrap, inputtime = inputtime)

train_DMrisk <- data.frame(Output = rep(c('DM'), nrow(train_DMrisk)), 
                          Dataset = rep(c('Train'), nrow(train_DMrisk)), 
                          train_DMrisk)
train_LRrisk <- data.frame(Output = rep(c('LR'), nrow(train_LRrisk)), 
                          Dataset = rep(c('Train'), nrow(train_LRrisk)), 
                          train_LRrisk)
inter_DMrisk <- data.frame(Output = rep(c('DM'), nrow(inter_DMrisk)), 
                          Dataset = rep(c('Inter'), nrow(inter_DMrisk)), 
                          inter_DMrisk)
inter_LRrisk <- data.frame(Output = rep(c('LR'), nrow(inter_LRrisk)), 
                          Dataset = rep(c('Inter'), nrow(inter_LRrisk)), 
                          inter_LRrisk)

exter_DMrisk <- data.frame(Output = rep(c('DM'), nrow(exter_DMrisk)), 
                          Dataset = rep(c('Exter'), nrow(exter_DMrisk)), 
                          exter_DMrisk)
exter_LRrisk <- data.frame(Output = rep(c('LR'), nrow(exter_LRrisk)), 
                          Dataset = rep(c('Exter'), nrow(exter_LRrisk)), 
                          exter_LRrisk)


save_datarisk <- rbind(train_DMrisk, inter_DMrisk, exter_DMrisk,
                   train_LRrisk, inter_LRrisk, exter_LRrisk)

################### radiomics output
train_DMrf <- boost_evaluation(train_data, 'DM', 'DM_Day', 'DM_rf', n_bootstrap = n_bootstrap, inputtime = inputtime)
train_LRrf <- boost_evaluation(train_data, 'LR', 'LR_Day', 'LR_rf', n_bootstrap = n_bootstrap, inputtime = inputtime)

inter_DMrf <- boost_evaluation(inter_data, 'DM', 'DM_Day', 'DM_rf', n_bootstrap = n_bootstrap, inputtime = inputtime)
inter_LRrf <- boost_evaluation(inter_data, 'LR', 'LR_Day', 'LR_rf', n_bootstrap = n_bootstrap, inputtime = inputtime)

exter_DMrf <- boost_evaluation(exter_data, 'DM', 'DM_Day', 'DM_rf', n_bootstrap = n_bootstrap, inputtime = inputtime)
exter_LRrf <- boost_evaluation(exter_data, 'LR', 'LR_Day', 'LR_rf', n_bootstrap = n_bootstrap, inputtime = inputtime)

train_DMrf <- data.frame(Output = rep(c('DM'), nrow(train_DMrf)), 
                           Dataset = rep(c('Train'), nrow(train_DMrf)), 
                           train_DMrf)
train_LRrf <- data.frame(Output = rep(c('LR'), nrow(train_LRrf)), 
                           Dataset = rep(c('Train'), nrow(train_LRrf)), 
                           train_LRrf)
inter_DMrf <- data.frame(Output = rep(c('DM'), nrow(inter_DMrf)), 
                           Dataset = rep(c('Inter'), nrow(inter_DMrf)), 
                           inter_DMrf)
inter_LRrf <- data.frame(Output = rep(c('LR'), nrow(inter_LRrf)), 
                           Dataset = rep(c('Inter'), nrow(inter_LRrf)), 
                           inter_LRrf)

exter_DMrf <- data.frame(Output = rep(c('DM'), nrow(exter_DMrf)), 
                           Dataset = rep(c('Exter'), nrow(exter_DMrf)), 
                           exter_DMrf)
exter_LRrf <- data.frame(Output = rep(c('LR'), nrow(exter_LRrf)), 
                           Dataset = rep(c('Exter'), nrow(exter_LRrf)), 
                           exter_LRrf)

save_datarf <- rbind(train_DMrf, inter_DMrf, exter_DMrf,
                       train_LRrf, inter_LRrf, exter_LRrf)

# 保存
library(openxlsx)
save_out = list('DL'=save_datarisk, 'RF'=save_datarf)
write.xlsx(save_out, file = file.path(save_dir, 'boost_output.xlsx'))



