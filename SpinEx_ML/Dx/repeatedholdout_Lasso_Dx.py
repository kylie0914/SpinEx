from sklearn import svm
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.linear_model import Lasso, LassoCV
from Utils import get_clf_result, save_data_2csv
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
import pickle

import warnings
warnings.filterwarnings("ignore")

np.random.seed(10)  # HK added this line
no_iteration = 1001 #  should be odd numbers... to find medain AUC .. (even number -> find average value)
split_seed = np.random.randint(10000, size=2000)  # HK_added this line: 1000 = iteration number


# ------ set parameters
norm_method =  "quantile"  # ['z-score', 'min-max', 'none', 'quantile']
selected_clf = "svm"  # svm, xgb
save_result = True
clf_cutoff = 0.5 #0.7234

svm_Cval= 10
svm_Gval =0.25

dataset_type =  {0: "all", 1:"early"}
selected_dataset = dataset_type[0]
print(selected_dataset)

save_path = "./Result/240703_repeat_result/Lasso_rbf_{0}/{3}_c{1}_g{2}".format(norm_method,svm_Cval,svm_Gval,selected_dataset)


# --- data load
feature_name = [' DCD', ' GFRA1', ' Her2', ' IGFR1', ' SMR3B', ' CEA',
                ' GPA33', ' MEP1A', ' SLC12A2', ' CD63', ' CD81', ' CD9',
                ' ASGR1', ' FGB', ' RBP4', ' SELENOP', ' c16orf89', ' DSG3',
                ' SCGB1A1', ' SP-B', ' Amy2A', ' GP2', ' REG1b', ' WNT2',
                ' CD24', ' EGFR', 'EpCAM', ' PDL-1', ' EGFRvIII', ' PDGFRalpha'] # , ' Age', ' Gender']

# ------ Data load: meta_data file contains tumor type, Dx label, Patient ID
if selected_dataset == dataset_type[0]:
	# --- median - all
	data_file = 'Data/features_median_ratio_exceptPA9_157.npy'
	meta_file = 'Data/patient_information_exceptPA9_157.csv'
if selected_dataset == dataset_type[1]:
	# --- median ealry 1,2, hd
	data_file = 'Data/EarlyStage_HD/Featrues_EarlyStage_HD_157.npy'
	meta_file = 'Data/EarlyStage_HD/Patient_info_EarlyStage_HD_157.csv'

test_acc_list = []

train_fpr_list = []
train_tpr_list = []
test_fpr_list = []
test_tpr_list = []
test_auc_list = []
train_auc_list = []

test_prediction_list = []
val_prediction_list = []
train_prediction_list = []
test_thr_list = []
meta_xtrain_list = []

min_test_auc = 1
max_test_auc = 0
min_fpr = []
min_tpr = []
min_prediction = []
min_thr = []
max_fpr = []
max_tpr = []
max_test_prediction = []

train_label_list =[]
test_label_list =[]

max_selected_test_data = None
max_test_label = None
max_clf = None


max_test_class_probs = []
max_train_class_probs = []
max_thr = []

min_index = 0
max_index = 0


features_data = np.load(data_file)
meta_data = pd.read_csv(meta_file)
meta_data.columns = ['Tumor', 'Dx', 'ID']

# Adding a new column for the tumor type
tumor_type =  ['HD', 'Br', 'Cl', 'Li', 'Lu', 'Pa'] # ['HD', 'Br', 'Lu', 'Cl', 'Pa', 'Li']# ['Br', 'Cl', 'Li', 'Lu', 'Pa']
for types in enumerate(tumor_type):
	meta_data.loc[meta_data['Tumor'] == types[1], 'Tumor_type'] = types[0]
# Converting feature data into a dataframe
# feature1 - the first dataset from all subjects
# feature2 - the second dataset from all subjects
features = pd.DataFrame(features_data, columns=feature_name)
feature_group1 = features[0::2];
feature_group2 = features[1::2];

# meta_group1 and meta_group2 should be the same in the current data structure
meta_group1 = meta_data[0::2]
meta_group2 = meta_data[1::2]

# label_data holds the Dx ground truths: 0/noncaner 1/cancer
labels = pd.DataFrame(np.array(meta_data['Dx'], dtype=np.int32))
label_group1 = labels[0::2]
label_group2 = labels[1::2]


if not os.path.exists(save_path):
	os.makedirs(save_path)


for iteration in tqdm(range(no_iteration)):
	if not os.path.exists(save_path + "/{0}/".format(iteration)):
		os.makedirs(save_path + "/{0}/".format(iteration))

	# ----------- 0. Preparing dataset
	# ---- 0-1) split dataset train&valid : test
	data_trainVal_ft1, data_test_ft1, label_trainVal_ft1, label_test_ft1, trainVal_id, test_id = \
		train_test_split(feature_group1, label_group1, range(len(feature_group1)),
		                 test_size=0.3, random_state= split_seed[iteration],  # 31 (svm), 11 (xgb)
		                 shuffle=True, stratify=meta_group1['Tumor_type'])
	# train&Valid
	data_trainVal = pd.concat([data_trainVal_ft1, feature_group2.iloc[trainVal_id]])
	label_trainVal = pd.concat([pd.DataFrame(label_trainVal_ft1), pd.DataFrame(label_group2.iloc[trainVal_id])])
	meta_trainVal = pd.concat([meta_group1.iloc[trainVal_id], meta_group2.iloc[trainVal_id]])

	# test
	data_test = pd.concat([data_test_ft1, feature_group2.iloc[test_id]])
	label_test = pd.concat([pd.DataFrame(label_test_ft1), pd.DataFrame(label_group2.iloc[test_id])])
	meta_test = pd.concat([meta_group1.iloc[test_id], meta_group2.iloc[test_id]])


	# print("----- [train] -----\n total size: {0}  \n each label : {1}".format(label_trainVal.size, sorted(label_trainVal.value_counts())))
	# print("------ Train Details (# of Data) -----\n", meta_trainVal["Tumor"].value_counts())
	# print("----- [test] -----\n total size: {0}   \n each label : {1}".format(label_test.size, sorted(label_test.value_counts())))
	# print("------ test Details (# of Data) -----\n", meta_test["Tumor"].value_counts())

	# ----------- 1. Pre-processing
	# ---- 1-1) Normalization
	if norm_method == 'z-score':
		scalar = StandardScaler()
		train_norm = scalar.fit_transform(data_trainVal)
		test_norm = scalar.transform(data_test)
	elif norm_method == 'min-max':
		scalar = MinMaxScaler(feature_range=(0, 10))
		scalar.fit(data_trainVal)
		train_norm = scalar.transform(data_trainVal)
		test_norm = scalar.transform(data_test)
	elif norm_method == 'quantile':
		scaler = QuantileTransformer()
		scaler.fit(data_trainVal)
		train_norm = scaler.transform(data_trainVal)
		test_norm = scaler.transform(data_test)
	elif norm_method == 'none':
		train_norm = np.array(data_trainVal)
		test_norm = np.array(data_test)


	# ----------- 2. Feature selection
	# ---- 2-1) Lasso feature selection
	lassocv = LassoCV()
	lassocv.fit(train_norm, label_trainVal)
	lassocv_alpha = lassocv.alpha_

	lasso = Lasso(alpha= lassocv_alpha)
	lasso.fit(train_norm, label_trainVal)
	train_score = lasso.score(train_norm, label_trainVal)  # Lasso score is R2 score.
	test_score = lasso.score(test_norm, label_test)

	# ---- 2-2) Selected features are corresponding to non-zero coefficients (weights).
	coeff_used = np.sum(lasso.coef_ != 0)
	ft_num = coeff_used

	# ---- 2-3) Compute accuracy of Lasso.
	pred_train = lasso.predict(train_norm)
	pred_train[abs(pred_train) < 0.5] = 0
	pred_train[abs(pred_train) >= 0.5] = 1

	pred_test= lasso.predict(test_norm)
	pred_test[abs(pred_test) < 0.5] = 0
	pred_test[abs(pred_test) >= 0.5] = 1

	np_train_label = np.array(label_trainVal)
	np_test_label = np.array(label_test)
	train_acc = (pred_train == np_train_label).mean()
	test_acc = (pred_test == np_test_label).mean()

	# print("# of features from Lasso: {0} (alpha value = {1})".format(coeff_used,lassocv_alpha))
	# print("[Lasso] train acc: {0} , valid acc: {1}, test acc: {2}".format(train_score, valid_score, test_score))

	# ---- 2-4)  Extract first n-rank features selected by absolute value of Lasso coefficient
	coeffs = lasso.coef_
	coeffs_rank = np.argsort(abs(coeffs))[::-1]
	select_rank = coeffs_rank[:ft_num]

	select_feature_train = np.array(train_norm[:, select_rank], dtype=np.float64)
	select_feature_test = np.array(test_norm[:, select_rank], dtype=np.float64)

	# Save Lasso coefficients of selected features
	select_feature_data = []
	select_feature_data.append(select_rank)
	select_feature_data.append(coeffs[select_rank])


	

	# ----------- 3. Classification
	# ---- 3-1) SVM classifier
	classifier = svm.SVC(C=svm_Cval, gamma=svm_Gval, kernel='rbf', probability=True, decision_function_shape='ovo', class_weight='balanced')
	classifier.fit(select_feature_train, np_train_label)
	train_score = classifier.score(select_feature_train, np_train_label)
	test_score = classifier.score(select_feature_test, np_test_label)

	train_class_probs = classifier.predict_proba(select_feature_train)
	test_class_probs = classifier.predict_proba(select_feature_test)

	train_cancer_probs = train_class_probs[:,1]
	test_cancer_probs =  test_class_probs[:, 1]

	# ---- 3-2) save classification results
	train_label_cancerProbs = np.column_stack( (np_train_label.reshape(-1,), train_class_probs[:,1]) )
	test_label_cancerProbs = np.column_stack((np_test_label.reshape(-1, ), test_class_probs[:, 1])) # test_class_probs[:, 1] = cancer probs
	train_HD_cancerProbs = train_label_cancerProbs[np.where(train_label_cancerProbs[:,0] ==0)]
	train_Cancer_cancerProbs = train_label_cancerProbs[np.where(train_label_cancerProbs[:,0] ==1)]

	test_HD_cancerProbs = test_label_cancerProbs[np.where(test_label_cancerProbs[:,0] ==0)]
	test_Cancer_cancerProbs = test_label_cancerProbs[np.where(test_label_cancerProbs[:, 0] == 1)]

	train_zipped_cancerProbs = pd.concat((pd.DataFrame(train_HD_cancerProbs), pd.DataFrame(train_Cancer_cancerProbs)), axis=1)
	train_zipped_cancerProbs.to_csv(save_path + "/{0}/Train_Cancer_probs.csv".format(iteration), header=["HD", "Cancer Probs", "Cancer", "Cancer Probs"])
	test_zipped_cancerProbs = pd.concat((pd.DataFrame(test_HD_cancerProbs), pd.DataFrame(test_Cancer_cancerProbs)), axis=1)
	test_zipped_cancerProbs.to_csv(save_path + "/{0}/Test_Cancer_probs.csv".format(iteration),
	                          header=["HD", "Cancer Probs", "Cancer", "Cancer Probs"])



	train_cancer_probs[abs(train_cancer_probs) < clf_cutoff] = 0
	train_cancer_probs[abs(train_cancer_probs) >= clf_cutoff] = 1


	test_cancer_probs[abs(test_cancer_probs) < clf_cutoff] = 0
	test_cancer_probs[abs(test_cancer_probs) >= clf_cutoff] = 1

	thr_train_score = accuracy_score(np_train_label, train_cancer_probs)
	thr_test_score =  accuracy_score(np_test_label,test_cancer_probs)

	train_cm, train_roc_auc, train_f1score, train_sensitivity, train_specificity, train_correct, train_fpr, train_tpr, train_thr, train_prediction = get_clf_result(
		classifier, select_feature_train, np_train_label)

	test_cm, test_roc_auc, test_f1score, test_sensitivity, test_specificity, test_correct, test_fpr, test_tpr, test_thr, test_prediction = get_clf_result(
		classifier, select_feature_test, np_test_label)

	test_acc_list.append(thr_test_score)


	train_fpr_list.append(train_fpr)
	train_tpr_list.append(train_tpr)
	test_fpr_list.append(test_fpr)
	test_tpr_list.append(test_tpr)

	train_label_list.append(np_train_label)
	test_label_list.append(np_test_label)

	train_auc_list.append(train_roc_auc)
	test_auc_list.append(test_roc_auc)

	train_prediction_list.append(train_prediction)
	test_prediction_list.append(test_prediction)

	test_thr_list.append(test_thr)

	if test_roc_auc > max_test_auc:
		max_train_auc = train_roc_auc
		max_test_auc = test_roc_auc
		max_fpr = test_fpr
		max_tpr = test_tpr

		max_test_prediction = test_prediction
		max_thr = test_thr
		max_index = iteration

		max_clf = classifier
		max_selected_test_data = select_feature_test
		max_test_label = np_test_label


		max_selected_train_data = select_feature_train
		max_train_label = np_train_label
		max_train_class_probs =train_class_probs
		max_test_class_probs = test_class_probs


	elif test_roc_auc < min_test_auc:
		min_test_auc = test_roc_auc
		min_fpr = test_fpr
		min_tpr = test_tpr
		min_prediction = test_prediction
		min_thr = test_thr
		min_index = iteration


	train_disp = ConfusionMatrixDisplay(train_cm).plot()
	plt.savefig(save_path + "/{1}/train_CW_CM_SVM_norm_{0}".format(norm_method, iteration))
	plt.close()

	test_disp = ConfusionMatrixDisplay(test_cm).plot()
	plt.savefig(save_path + "/{1}/test_CW_CM_SVM_norm_{0}".format(norm_method, iteration))
	plt.close()


	# Save iterations records

	records = []
	records.append(['Iter_No', ' Lasso_No', ' Select_No', ' Lasso_train',
	                ' Lasso_test', ' SVM_train', ' SVM_test', 'thr_train_score','Sensitivity', 'Specificity', 'F1score', 'AUC'])
	records.append([iteration, coeff_used, ft_num, train_acc,
	                test_acc, train_score, test_score, thr_train_score,train_sensitivity, train_specificity, train_f1score, train_roc_auc])
	save_data_2csv(save_path + '/{0}/train_svm_result.csv'.format(iteration), records)


	records2 = []
	records2.append(['Iter_No', ' Lasso_No', ' Select_No', ' Lasso_train',
	                 ' Lasso_test', ' SVM_train', ' SVM_test', 'thr_test_score','Sensitivity', 'Specificity', 'F1score', 'AUC'])
	records2.append([iteration, coeff_used, ft_num, train_acc,
	                 test_acc, train_score, test_score,thr_test_score, test_sensitivity, test_specificity, test_f1score, test_roc_auc])
	save_data_2csv(save_path + '/{0}/test_svm_result.csv'.format(iteration), records2)



		



# -----------4. validation results
# ---- 4-1) Mean accuracy and AUC for 1000 times repeated holdout
train_auc_list = np.array(train_auc_list, dtype=np.float64)
test_auc_list = np.array(test_auc_list, dtype=np.float64)
test_acc_liist = np.array(test_acc_list, dtype = np.float64)

print(" ------ AUC summary ------ \n\t Max {0} \t Min AUC: {1} \t Avg. AUC: {2} ".format(np.max(test_auc_list), np.min(test_auc_list), np.mean(test_auc_list)))
print(" ------ ACC summary ------ \n\t Max {0} \t Min AUC: {1} \t Avg. AUC: {2} ".format(np.max(test_acc_liist), np.min(test_acc_liist), np.mean(test_acc_liist)))
max_auc = np.max(test_auc_list)


median_train_auc = np.median(train_auc_list)
median_test_auc = np.median(test_auc_list)
median_index = np.where(test_auc_list == median_test_auc)[0][0]

# --- train tpr, fpr for best model
max_train_fpr = train_fpr_list[max_index]
max_train_tpr = train_tpr_list[max_index]

# --- test tpr, fpr for best model
max_test_fpr = test_fpr_list[max_index]
max_test_tpr = test_tpr_list[max_index]
min_test_fpr = test_fpr_list[min_index]
min_test_tpr = test_tpr_list[min_index]

# --- test tpr, fpr for median model
median_test_fpr = test_fpr_list[median_index]
median_test_tpr = test_tpr_list[median_index]

median_test_thr = test_thr_list[median_index]


median_train_fpr = train_fpr_list[median_index]
median_train_tpr = train_tpr_list[median_index]

max_train_class_probs = train_prediction_list[max_index]

median_train_class_probs = train_prediction_list[median_index]
median_train_label = train_label_list[median_index]

max_test_class_probs = test_prediction_list[max_index]
median_test_class_probs = test_prediction_list[median_index]
median_test_label = test_label_list[median_index]


# ---- 4-2) Save best model and results
if not os.path.exists(save_path + "/records/"):
	os.makedirs(save_path + "/records/")

with open(save_path +'/records/best_model.pickle','wb') as fw:
    pickle.dump(max_clf, fw)


save_data_2csv(save_path + '/records/train_selected_features_best_model.csv',max_selected_train_data)
save_data_2csv(save_path + '/records/train_label_best_model.csv',max_train_label)
save_data_2csv(save_path + '/records/train_predictions_best_model.csv',max_train_class_probs)


save_data_2csv(save_path + '/records/test_selected_features_best_model.csv',max_selected_test_data)
save_data_2csv(save_path + '/records/test_label_best_model.csv',max_test_label)
save_data_2csv(save_path + '/records/test_predictions_best_model.csv',max_test_class_probs)


save_data_2csv(save_path + '/records/train_predictions_median_model.csv',median_train_class_probs)
save_data_2csv(save_path + '/records/train_label_median_model.csv',median_train_label)

save_data_2csv(save_path + '/records/test_predictions_median_model.csv',median_test_class_probs)
save_data_2csv(save_path + '/records/test_label_median_model.csv',median_test_label)

# ---- 4-3) median model AUC plot
plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(max_test_fpr, max_test_tpr, '-r', label='test (area = {:.3f})'.format(max_test_auc))
plt.plot(median_test_fpr, median_test_tpr, '-r', label='test (area = {:.3f})'.format(median_test_auc))
plt.plot(median_train_fpr, median_train_tpr, '-b', label='training (area = {:.3f})'.format(median_train_auc))

plt.xlabel('1-Sensitivity', fontsize=14)
plt.ylabel('Sensitivity', fontsize=14)
plt.title('ROC curve for median model', fontsize=14)
plt.legend(loc='median')
plt.savefig(save_path + "/records/median_mediandata_AUC_ROC_clf_{0}_norm_{1}_iteration_{2}".format("svm", norm_method, median_index))
plt.show()

# ---- 4-4) best model AUC plot
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(max_test_fpr, max_test_tpr, '-r', label='test (area = {:.3f})'.format(max_test_auc))
# plt.plot(median_test_fpr, median_test_tpr, '-r', label='test (area = {:.3f})'.format(median_test_auc))
plt.plot(max_train_fpr, max_train_tpr, '-b', label='training (area = {:.3f})'.format(max_train_auc))

plt.xlabel('1-Sensitivity', fontsize=14)
plt.ylabel('Sensitivity', fontsize=14)
plt.title('ROC curve for best model', fontsize=14)
plt.legend(loc='best')
plt.savefig(save_path + "/records/best_model_mediandata_AUC_ROC_clf_{0}_norm_{1}_iteration_{2}".format("svm", norm_method, max_index))
plt.show()









