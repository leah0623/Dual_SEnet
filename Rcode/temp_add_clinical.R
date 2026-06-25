# 主要任务 利用临床数据构建DM和LR对比模型
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

cdata <- read_excel('D:/project/0_Competition/data/new_data/Clinical.xlsx', sheet='new')
cdata$allstage <- sapply(cdata$TNMstage, function(x) ifelse((x=='I' || x=='II'), 'IandII', x))
cdata$allstage <- factor(cdata$allstage , levels = c('IandII', 'III', 'IV'))

cdata$dataset <- factor(cdata$dataset, levels = c('Train', 'Inter', 'Exter'))
cdata$Group <- factor(cdata$Group, levels= c('HGJ', 'CHUS', 'CHUM', 'HMR'))
cdata$Sex <- factor(cdata$Sex, levels = c('F', 'M'))
cdata$Primary_Site <- factor(cdata$Primary_Site , levels = c('Oropharynx', 'Hypopharynx', 'Nasopharynx', 'Larynx'))
cdata$TNMstage <- factor(cdata$TNMstage, levels = c('I', 'II', 'III', 'IV'))
cdata$T_stage <- factor(cdata$T_stage, levels = c('T1', 'T2', 'T3', 'T4'))
cdata$N_stage <- factor(cdata$N_stage, levels = c('N0', 'N1', 'N2', 'N3'))
cdata$Therapy <- factor(cdata$Therapy, levels = c('chemo radiation', 'radiation'))

# load dl score
outscore <- read_excel('D:/project/0_Competition/data/new_data/DL_RF.xlsx')
finaldata <- cbind(cdata, outscore[c(6,10)])

save_dir <- 'D:/project/0_Competition/data/new_data/add_clinical'

train_data <- finaldata[finaldata$dataset == 'Train',]
inter_data <- finaldata[finaldata$dataset == 'Inter',]
exter_data <- finaldata[finaldata$dataset == 'Exter',]
valid_data <- finaldata[finaldata$dataset != 'Train',]

# 将年龄进行归一
preProValues <- preProcess(train_data[c(5)],method = c("center","scale"))
train_data[c(5)] <- predict(preProValues, train_data[c(5)])
inter_data[c(5)] <- predict(preProValues, inter_data[c(5)])
exter_data[c(5)] <- predict(preProValues, exter_data[c(5)])
valid_data[c(5)] <- predict(preProValues, valid_data[c(5)])
################
varName <- c('Sex', 'Age', 'Primary_Site', 'T_stage', 'N_stage', 'allstage', 'Therapy')
new_data <- cbind(train_data[c('DM_Day', 'Distant', 'LR_Day', 'Locoregional')], train_data[varName]) %>% data.frame()

############
# base function
ucox_fit <- function(event, event_time, varnamei, basedata){
  cox_f <- as.formula(paste('Surv(', event_time, ',' , event,') ~ -1 + ', varnamei))
  coxmodel <- coxph(cox_f, data=up_train)
  coxsum <- summary(coxmodel)
  coxcoef <- coef(coxsum)
  
  HR <- coxcoef[, 2]
  SE <- coxcoef[, 3]
  CI5 <- HR-1.96*SE
  CI95 <- HR+1.96*SE
  pvalue <- coxcoef[, 5]
  uni_cox_out <- data.frame('Feature'= varnamei, 
                            'HR'=HR, 'CI5' = CI5, 
                            'CI95' = CI95, 'Pvalue' = pvalue)
  return(uni_cox_out)
}

# 5-flod
cv_fit <- function(formla, event, event_day, dataset){
  set.seed(123)
  folds <- createFolds(y=dataset[[event]],k=5)
  ci_valid <- vector(mode="numeric",length=0)
  AUC_valid <- vector(mode="numeric",length=0)
  model_list <- list()
  for (i in 1:5) {
    val_data_i <- dataset[folds[[i]],]  #取folds[[j]]作为测试集
    tra_data_i <- dataset[-folds[[i]],]  # 剩下的数据作为训练集
    
    # 重采样
    up_train <- upSample(x = tra_data_i, y = as.factor(tra_data_i[[event]]))
    model_CDD <- coxph(formla, data=up_train)
    fp1 <- predict(model_CDD, val_data_i)
    # C-index
    ci_valid[i] <- 1-rcorr.cens(fp1,Surv(val_data_i[[event_day]],val_data_i[[event]])) [[1]]
    ROC_valid <- timeROC(T=val_data_i[[event_day]],delta=val_data_i[[event]],marker=fp1,
                         cause=1,weighting="marginal",times=c(1800),ROC = TRUE,iid = FALSE)
    AUC_valid[i] <- as.numeric(ROC_valid$AUC[2])
    
    model_list[[i]] <- model_CDD
  }
  parameter_status <- data.frame(ci_valid,AUC_valid)
  model_num <- which.max(parameter_status[,1])
  best_fit <- model_list[[model_num]]
  return(list(bf=best_fit, ml=model_list))
}
############
# 重采样平衡数据
set.seed(123)
up_train <- upSample(x = new_data, y = as.factor(new_data[['Locoregional']]))

#### LR
# 单因素Cox选择临床特征
uni_outLR <- data.frame('Feature'='x', 'HR'=0, 'CI5' = 0, 'CI95' = 0, 'Pvalue' = 0)
for (i in 1:length(varName)){
  uni_out <- ucox_fit('Locoregional', 'LR_Day', varName[i], up_train)
  uni_outLR <- rbind(uni_outLR, uni_out)
}
uni_outLR <- uni_outLR[-1,]
uni_outLR$Feature_grade <- rownames(uni_outLR)

select_clinical <- unique(uni_outLR$Feature[uni_outLR$Pvalue < 0.05])

# 五折交叉建立模型
fit_step <- as.formula(paste('Surv(LR_Day, Locoregional) ~',paste(c(select_clinical, 'LR_risk'), collapse = '+')))
clinical_LRfit <- cv_fit(fit_step, 'Locoregional', 'LR_Day', train_data)
ci_all <- matrix(0, nrow=3, ncol=1)
fp_train <- predict(clinical_LRfit$bf, train_data)
fp_inter <- predict(clinical_LRfit$bf, inter_data)
fp_exter <- predict(clinical_LRfit$bf, exter_data)

ci_all[1,1] <- 1-rcorr.cens(fp_train, Surv(train_data$LR_Day, train_data$Locoregional))[1] %>% as.numeric()
ci_all[2,1] <- 1-rcorr.cens(fp_inter, Surv(inter_data$LR_Day, inter_data$Locoregional))[1] %>% as.numeric()
ci_all[3,1] <- 1-rcorr.cens(fp_exter, Surv(exter_data$LR_Day, exter_data$Locoregional))[1] %>% as.numeric()


#### DM
set.seed(123)
up_train <- upSample(x = new_data, y = as.factor(new_data[['Distant']]))
# 单因素Cox选择临床特征
uni_outDM <- data.frame('Feature'='x', 'HR'=0, 'CI5' = 0, 'CI95' = 0, 'Pvalue' = 0)
for (i in 1:length(varName)){
  uni_out <- ucox_fit('Distant', 'DM_Day', varName[i], up_train)
  uni_outDM <- rbind(uni_outDM, uni_out)
}
uni_outDM <- uni_outDM[-1,]
uni_outDM$Feature_grade <- rownames(uni_outDM)

select_clinical <- unique(uni_outDM$Feature[uni_outDM$Pvalue < 0.05])
select_clinical <- c("Sex", "Primary_Site","Therapy", "allstage")

# 五折交叉建立模型
fit_step <- as.formula(paste('Surv(DM_Day, Distant) ~',paste(c(select_clinical, 'DM_risk'), collapse = '+')))
clinical_DMfit <- cv_fit(fit_step, 'Distant', 'DM_Day', train_data)
ci_all <- matrix(0, nrow=3, ncol=1)
fp_train <- predict(clinical_DMfit$bf, train_data)
fp_inter <- predict(clinical_DMfit$bf, inter_data)
fp_exter <- predict(clinical_DMfit$bf, exter_data)

ci_all[1,1] <- 1-rcorr.cens(fp_train, Surv(train_data$DM_Day, train_data$Distant))[1] %>% as.numeric()
ci_all[2,1] <- 1-rcorr.cens(fp_inter, Surv(inter_data$DM_Day, inter_data$Distant))[1] %>% as.numeric()
ci_all[3,1] <- 1-rcorr.cens(fp_exter, Surv(exter_data$DM_Day, exter_data$Distant))[1] %>% as.numeric()

### add
coxsum <- summary(clinical_DMfit$ml[[1]])
coxcoef <- coef(coxsum)

HR <- coxcoef[, 2]
SE <- coxcoef[, 3]
CI5 <- HR-1.96*SE
CI95 <- HR+1.96*SE
pvalue <- coxcoef[, 5]
uni_cox_out <- data.frame('HR'=HR, 'CI5' = CI5, 
                          'CI95' = CI95, 'Pvalue' = pvalue)


############ savedata
library(openxlsx)
# predict
all_result <- finaldata
all_result$DMadd <- predict(clinical_DMfit$bf, finaldata)
all_result$LRadd <- predict(clinical_LRfit$bf, finaldata)
write.xlsx(all_result, file.path(save_dir, 'addresults_0327.xlsx'))

allsave = list(DMun=uni_outDM, LRun=uni_outLR)
write.xlsx(allsave, file.path(save_dir, 'clinicalUN_0327.xlsx'))
