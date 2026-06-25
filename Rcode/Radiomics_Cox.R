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

base_dir <- 'D:/project/0_Competition/data/new_data'
clinical_path <- file.path(base_dir, 'ALL_dm_lr.xlsx')
radiomics_path <- file.path(base_dir, 'radiomics_0828.xlsx')

CHUS_label <- read_excel(clinical_path, sheet='CHUS') # all_me.xlsx
HGJ_label <- read_excel(clinical_path, sheet='HGJ')
CHUM_label <- read_excel(clinical_path, sheet='CHUM')
HMR_label <- read_excel(clinical_path, sheet='HMR')

CHUS_RF <- read_excel(radiomics_path, sheet='CHUS')
HGJ_RF <- read_excel(radiomics_path, sheet='HGJ')
CHUM_RF <- read_excel(radiomics_path, sheet='CHUM')
HMR_RF <- read_excel(radiomics_path, sheet='HMR')

CHUS <- cbind(CHUS_label, CHUS_RF[-c(1)])
HGJ <- cbind(HGJ_label, HGJ_RF[-c(1)])
CHUM <- cbind(CHUM_label, CHUM_RF[-c(1)])
HMR <- cbind(HMR_label, HMR_RF[-c(1)])

train_data <- rbind(CHUS, HGJ)
val_data <- CHUM
test_data <- HMR
train_data <- na.omit(train_data) %>% data.frame()
val_data <- na.omit(val_data) %>% data.frame()
test_data <- na.omit(test_data) %>% data.frame()

preProValues <- preProcess(train_data[c(6:1923)],method = c("center","scale"))
train_features <- predict(preProValues, train_data[c(6:1923)])
val_features <- predict(preProValues, val_data[c(6:1923)])
test_features <- predict(preProValues, test_data[c(6:1923)])

train_features <- data.frame(train_features)
val_features <- data.frame(val_features)
test_features <- data.frame(test_features)

##########################
# DM
##########################
train_label <- train_data[c(2,3)]
DMweight <- numeric(nrow(train_label))
DMweight[train_label$DM == 0] <- rep(1, sum(train_label$DM == 0))
DMweight[train_label$DM == 1] <- rep(1, sum(train_label$DM == 1))

train_features_up <- train_features
trainCor <- cor(train_features_up, method = "pearson")
highCorDescr <- findCorrelation(trainCor,cutoff = 0.85,names = F,verbose = F,exact = F)
DM.pccfeatures <- train_features_up[,-highCorDescr]

varName <- colnames(DM.pccfeatures)
new_data <- cbind(train_label, DM.pccfeatures) %>% data.frame()
univ_formules <- sapply(varName, function(x) as.formula(paste('Surv(DM_Day, DM)~', x)))
univ_models <- lapply(univ_formules, function(x){coxph(x, data = new_data, weights = DMweight)})
univ_results <- lapply(univ_models, function(x){
  x <- summary(x)
  pvalue <- x$wald['pvalue']
  return(pvalue)
})
DM.uncoxfeatures <- DM.pccfeatures[(univ_results<0.05)]

x <- as.matrix(DM.uncoxfeatures)
y <- Surv(train_label$DM_Day, train_label$DM)
fit <- glmnet(x, y, alpha=1,family = 'cox')
plot(fit, xvar = "lambda", label = TRUE,lwd=2)
set.seed(123)
fit_cv <- cv.glmnet(x, y, alpha=1, family = 'cox',type.measure = 'C', nfolds=5, weights = DMweight)
plot(fit_cv,lwd=2)
Coefficients <- coef(fit_cv, s = fit_cv$lambda.min)
Active.Index <- which(Coefficients != 0)
Active.Coefficients <- Coefficients[Active.Index]
varname <- rownames(Coefficients)[Active.Index]
number <- vector(mode="numeric",length=0)
for (i in 1:length(varname)) {
  if(varname[i]!="(Intercept)"){
    number[i] <- which(colnames(DM.uncoxfeatures)==varname[i])
  }
}
DM.muCoxfeatures <- DM.uncoxfeatures[number]

set.seed(123)
DM_step <- as.formula(paste('Surv(DM_Day, DM) ~',paste(colnames(DM.muCoxfeatures), collapse = '+')))
train_final_data <- cbind(train_label, DM.muCoxfeatures) %>% data.frame()
folds <- createFolds(y=train_final_data$DM,k=5)
ci_valid <- vector(mode="numeric",length=0)
AUC_valid <- vector(mode="numeric",length=0)
model_list <- list()
for (i in 1:5) {
  val_data_i <- train_final_data[folds[[i]],]  #取folds[[j]]作为测试集
  tra_data_i <- train_final_data[-folds[[i]],]  # 剩下的数据作为训练集
  
  # 重采样
  up_train <- upSample(x = tra_data_i, y = as.factor(tra_data_i$DM))
  model_CDD <- cph(DM_step, x=T, y=T, surv=T, data=up_train, time.inc = c(1800))
  
  fp1 <- predict(model_CDD, val_data_i)
  # C-index
  ci_valid[i] <- 1-rcorr.cens(fp1,Surv(val_data_i$DM_Day,val_data_i$DM)) [[1]]
  
  # AUC
  ROC_valid <- timeROC(T=val_data_i$DM_Day,delta=val_data_i$DM,marker=fp1,
                       cause=1,weighting="marginal",times=c(1800),ROC = TRUE,iid = FALSE)
  AUC_valid[i] <- as.numeric(ROC_valid$AUC[2])
  model_list[[i]] <- model_CDD
}
parameter_status <- data.frame(ci_valid,AUC_valid)
model_num <- which.max(parameter_status[,1])
DM.RFfit <- model_list[[model_num]]

fp_DMRF_train <- predict(DM.RFfit, train_features)
fp_DMRF_valid <- predict(DM.RFfit, val_features)
fp_DMRF_test <- predict(DM.RFfit, test_features)

rowname_list <- c('train', 'valid', 'test')
colname_list <- c('DMRF', 'LRRF')
ci_all <- matrix(0, nrow=3, ncol=2)
ci_all[1,1] <- 1-rcorr.cens(fp_DMRF_train, Surv(train_data$DM_Day, train_data$DM))[1] %>% as.numeric()
ci_all[2,1] <- 1-rcorr.cens(fp_DMRF_valid, Surv(val_data$DM_Day, val_data$DM))[1] %>% as.numeric()
ci_all[3,1] <- 1-rcorr.cens(fp_DMRF_test, Surv(test_data$DM_Day, test_data$DM))[1] %>% as.numeric()
ci_all

##########################
# LR
##########################
train_label <- train_data[c(4,5)]
LRweight <- numeric(nrow(train_label))
LRweight[train_label$LR == 0] <- rep(1, sum(train_label$LR == 0))
LRweight[train_label$LR == 1] <- rep(1, sum(train_label$LR == 1))
trainCor <- cor(train_features, method = "pearson")
highCorDescr <- findCorrelation(trainCor,cutoff = 0.85,names = F,verbose = F,exact = F)
LR.pccfeatures <- train_features[,-highCorDescr]

varName <- colnames(LR.pccfeatures)
new_data <- cbind(train_label, LR.pccfeatures) %>% data.frame()
univ_formules <- sapply(varName, function(x) as.formula(paste('Surv(LR_Day, LR)~', x)))
univ_models <- lapply(univ_formules, function(x){coxph(x, data = new_data, weights = LRweight)})
univ_results <- lapply(univ_models, function(x){
  x <- summary(x)
  pvalue <- x$wald['pvalue']
  return(pvalue)
})
LR.uncoxfeatures <- LR.pccfeatures[(univ_results<0.05)]

x <- as.matrix(LR.uncoxfeatures)
y <- Surv(train_label$LR_Day, train_label$LR)
fit <- glmnet(x, y, alpha=1,family = 'cox')
plot(fit, xvar = "lambda", label = TRUE,lwd=2)
set.seed(123)
fit_cv <- cv.glmnet(x, y, alpha=1, family = 'cox',type.measure = 'C', nfolds=5, weights = LRweight)
plot(fit_cv,lwd=2)
Coefficients <- coef(fit_cv, s = fit_cv$lambda.min)
Active.Index <- which(Coefficients != 0)
Active.Coefficients <- Coefficients[Active.Index]
varname <- rownames(Coefficients)[Active.Index]
number <- vector(mode="numeric",length=0)
for (i in 1:length(varname)) {
  if(varname[i]!="(Intercept)"){
    number[i] <- which(colnames(LR.uncoxfeatures)==varname[i])
  }
}
LR.muCoxfeatures <- LR.uncoxfeatures[number]

set.seed(123)
LR_step <- as.formula(paste('Surv(LR_Day, LR) ~',paste(colnames(LR.muCoxfeatures), collapse = '+')))
train_final_data <- cbind(train_label, LR.muCoxfeatures) %>% data.frame()
lrfolds <- createFolds(y=train_final_data$LR,k=5)
ci_valid <- vector(mode="numeric",length=0)
AUC_valid <- vector(mode="numeric",length=0)
model_list <- list()
for (i in 1:5) {
  val_data_i <- train_final_data[lrfolds[[i]],]  #取folds[[j]]作为测试集
  tra_data_i <- train_final_data[-lrfolds[[i]],]  # 剩下的数据作为训练集
  
  # 重采样
  up_train <- upSample(x = tra_data_i, y = as.factor(tra_data_i$LR))
  model_CDD <- cph(LR_step, x=T, y=T, surv=T, data=up_train, time.inc = c(1800))
  
  fp1 <- predict(model_CDD, val_data_i)
  # C-index
  ci_valid[i] <- 1-rcorr.cens(fp1,Surv(val_data_i$LR_Day,val_data_i$LR)) [[1]]
  
  # AUC
  ROC_valid <- timeROC(T=val_data_i$LR_Day,delta=val_data_i$LR,marker=fp1,
                       cause=1,weighting="marginal",times=c(1800),ROC = TRUE,iid = FALSE)
  AUC_valid[i] <- as.numeric(ROC_valid$AUC[2])
  model_list[[i]] <- model_CDD
}
parameter_status <- data.frame(ci_valid,AUC_valid)
model_num <- which.max(parameter_status[,1])
LR.RFfit <- model_list[[model_num]]

fp_LRRF_train <- predict(LR.RFfit, train_features)
fp_LRRF_valid <- predict(LR.RFfit, val_features)
fp_LRRF_test <- predict(LR.RFfit, test_features)

ci_all[1,2] <- 1-rcorr.cens(fp_LRRF_train, Surv(train_data$LR_Day, train_data$LR))[1] %>% as.numeric()
ci_all[2,2] <- 1-rcorr.cens(fp_LRRF_valid, Surv(val_data$LR_Day, val_data$LR))[1] %>% as.numeric()
ci_all[3,2] <- 1-rcorr.cens(fp_LRRF_test, Surv(test_data$LR_Day, test_data$LR))[1] %>% as.numeric()
ci_all

##########################
# savedata
##########################
library(openxlsx)
# predict
all_features <- rbind(train_features, val_features, test_features)
all_result <- rbind(train_data[c(1:5)], val_data[c(1:5)], test_data[c(1:5)])
all_result$DMRF <- predict(DM.RFfit, all_features)
all_result$LRRF <- predict(LR.RFfit, all_features)

# features
DM_names <- colnames(DM.muCoxfeatures)
final_DM_select <- data.frame(
  label = paste0("DM", seq_along(DM_names)),
  column_name = DM_names,
  stringsAsFactors = FALSE
)
LR_names <- colnames(LR.muCoxfeatures)
final_LR_select <- data.frame(
  label = paste0("LR", seq_along(LR_names)),
  column_name = LR_names,
  stringsAsFactors = FALSE
) 
all_select_name <- rbind(final_DM_select, final_LR_select)

DMsum <- coef(DM.RFfit)
DM_names <- as.data.frame(DMsum) %>% rownames() 
final_DM_select <- data.frame(
  label = paste0("DM", seq_along(DM_names)),
  column_name = DM_names,
  coef = DMsum,
  stringsAsFactors = FALSE
)

LRsum <- coef(LR.RFfit)
LR_names <- as.data.frame(LRsum) %>% rownames() 
final_LR_select <- data.frame(
  label = paste0("LR", seq_along(LR_names)),
  column_name = LR_names,
  coef = LRsum,
  stringsAsFactors = FALSE
)
all_select_name <- rbind(final_DM_select, final_LR_select)

# num
DM_process <- data.frame(
  process = c('DM_pcc', 'DM_UnCox', 'DM_LASSO'),
  nums = c(length(DM.pccfeatures),length(DM.uncoxfeatures), length(DM.muCoxfeatures)),
  stringsAsFactors = FALSE
)
LR_process <- data.frame(
  process = c('LR_pcc', 'LR_UnCox', 'LR_LASSO'),
  nums = c(length(LR.pccfeatures), length(LR.uncoxfeatures), length(LR.muCoxfeatures)),
  stringsAsFactors = FALSE
)
all_process <- rbind(DM_process, LR_process)

allsave <- list(
  'CI' = ci_all, 'PredictOut' = all_result, 
  'SelectFeatures' = all_select_name,
  'SelectProcess' =  all_process
)
write.xlsx(allsave, file.path(base_dir, 'output_coxRadiomics_0327.xlsx'))




