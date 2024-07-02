from sklearn import svm
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.model_selection import StratifiedKFold
from Utils import save_data_2csv, get_clf_result

from sklearn import metrics

import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import os
import seaborn as sns

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.linear_model import  LogisticRegressionCV

import warnings
warnings.filterwarnings("ignore")

np.random.seed(30)  # HK added this line
split_seed = np.random.randint(10000, size=2000)  # HK_added this line: 1000 = iteration number


# -------- 0. Set parameters ----------
norm_method = "z-score" #  ['z-score', 'min-max','quantile', 'none']
selected_clf = "svm"
save_result = True


dataset_type =  {0: "early", 1:"all"} # early : HD vs Cancer stage 1,2 ,  all : HD vs All cancer stage (1,2,3,4)
selected_dataset = dataset_type[1]


svm_Cval = 10
svm_Gval = 1.2


feature_name = [' DCD', ' GFRA1', ' Her2', ' IGFR1', ' SMR3B', ' CEA',
                ' GPA33', ' MEP1A', ' SLC12A2', ' CD63', ' CD81', ' CD9',
                ' ASGR1', ' FGB', ' RBP4', ' SELENOP', ' c16orf89', ' DSG3',
                ' SCGB1A1', ' SP-B', ' Amy2A', ' GP2', ' REG1b', ' WNT2',
                ' CD24', ' EGFR', 'EpCAM', ' PDL-1', ' EGFRvIII', ' PDGFRalpha']




if selected_dataset == dataset_type[0]:
	# --- median - all
	data_file = 'Data/features_median_ratio_exceptPA9.npy'
	meta_file = 'Data/patient_information_exceptPA9.csv'
if selected_dataset == dataset_type[1]:
	# --- median ealry 1,2, hd
	data_file = 'Data/EarlyStage_HD/Featrues_EarlyStage_HD.npy'
	meta_file = 'Data/EarlyStage_HD/Patient_info_EarlyStage_HD_only_C.csv'


features_data = np.load(data_file)
meta_data = pd.read_csv(meta_file)
meta_data.columns = ['Tumor', 'Dx', 'ID']

# Adding a new column for the tumor type
tumor_type = ['HD', 'Br', 'Lu', 'Cl', 'Pa', 'Li']
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

fpr_list = []
tpr_list = []
auc_list = []
prediction_list = []

max_clf = None
max_cm = None
min_auc = 1
max_auc = 0
min_fpr = []
min_tpr = []
max_fpr = []
max_tpr = []

min_index = 0
max_index = 0

max_selected_test_data = None
max_selected_test_label = None

fold_num = 5  # --- 전체 데이터  5-fold
fold_cnt = 0

avg_acc = {'train': [], 'test': []}
avg_f1score = {'train': [], 'test': []}
coef_list = []

save_path = "./Result/Fold_Dataset_" + selected_dataset + "_results/logreg_5fold/c{0}_g{1}/".format(svm_Cval,svm_Gval)
if not os.path.exists(save_path):
	os.makedirs(save_path)

losses = []
kf = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=31)

for fold_train_idx, fold_test_idx in kf.split(feature_group1, meta_group1['Dx']):
	# ----------0. Preparing dataset
	print("----- [Each fold result] {0} fold result -----".format(fold_cnt))
	train_data_feature1 = pd.DataFrame([])
	train_data_feature2 = pd.DataFrame([])
	test_data_feature1 = pd.DataFrame([])
	test_data_feature2 = pd.DataFrame([])

	train_label_feature1 = pd.DataFrame([])
	train_label_feature2 = pd.DataFrame([])
	test_label_feature1 = pd.DataFrame([])
	test_label_feature2 = pd.DataFrame([])

	# ---- 0-1) train, test feature data
	train_data_feature1 = pd.concat([train_data_feature1, feature_group1.iloc[fold_train_idx]])
	train_data_feature2 = pd.concat([train_data_feature2, feature_group2.iloc[fold_train_idx]])
	train_data_fin = pd.concat([train_data_feature1, train_data_feature2])

	test_data_feature1 = pd.concat([test_data_feature1, feature_group1.iloc[fold_test_idx]])
	test_data_feature2 = pd.concat([test_data_feature2, feature_group2.iloc[fold_test_idx]])
	test_data_fin = pd.concat([test_data_feature1, test_data_feature2])

	# ---- 0-2) train, test label data
	train_label_feature1 = pd.concat([train_label_feature1, label_group1.iloc[fold_train_idx]])
	train_label_feature2 = pd.concat([train_label_feature2, label_group2.iloc[fold_train_idx]])
	train_label_fin = pd.concat([train_label_feature1, train_label_feature2])

	test_label_feature1 = pd.concat([test_label_feature1, label_group1.iloc[fold_test_idx]])
	test_label_feature2 = pd.concat([test_label_feature2, label_group2.iloc[fold_test_idx]])
	test_label_fin = pd.concat([test_label_feature1, test_label_feature2])


	# ---- 0-3) Save train/test dataset for each fold
	if save_result:
		if not os.path.exists(save_path + "/fold_result/"):
			os.makedirs(save_path + "/fold_result/")
		# train_data_fin.to_csv(save_path + "/fold_result/fold{0}_train_data.csv".format(fold_cnt),
		#                       index_label='sample_ID')
		# test_data_fin.to_csv(save_path + "/fold_result/fold{0}_test_data.csv".format(fold_cnt), index_label='sample_ID')
		# train_label_fin.to_csv(save_path + "/fold_result/fold{0}_train_label.csv".format(fold_cnt),
		#                        index_label='sample_ID')
		# test_label_fin.to_csv(save_path + "/fold_result/fold{0}_test_label.csv".format(fold_cnt),
		#                       index_label='sample_ID')
		# 
		# print("----- [train] -----\n total size: {0}  \n each label : {1}".format(train_label_fin.size, train_label_fin.value_counts()))
		# print("----- [test] -----\n Ωtotal size: {0}   \n each label : {1}".format(test_label_fin.size, test_label_fin.value_counts()))


	# ---------- 1. Normalize

	if norm_method == 'z-score':
		scalar = StandardScaler()
		scalar.fit(train_data_fin)
		train_norm = scalar.fit_transform(train_data_fin)
		test_norm = scalar.transform(test_data_fin)
	elif norm_method == 'min-max':
		scalar = MinMaxScaler(feature_range=(0, 10))
		scalar.fit(train_data_fin)
		train_norm = scalar.transform(train_data_fin)
		test_norm = scalar.transform(test_data_fin)
	elif norm_method == 'quantile':
		scaler = QuantileTransformer()
		scaler.fit(train_data_fin)
		train_norm = scaler.transform(train_data_fin)
		test_norm = scaler.transform(test_data_fin)
	elif norm_method == 'none':
		train_norm = np.array(train_data_fin)
		test_norm = np.array(test_data_fin)

	np_train_label = np.array(train_label_fin)
	np_test_label = np.array(test_label_fin)

	# ---------- 2. Feature selection
	# ---- 2-1) logistic regression

	Logreg_cs = [0.01, 0.1, 0.5, 0.8, 1, 1.2, 2,3,  5, 10, 30, 50, 100]
	Logreg_cv = StratifiedKFold(n_splits=fold_num, shuffle=False)
	Logistic_model = LogisticRegressionCV(
		Cs=Logreg_cs,  # 10개의 다른 C 값 테스
		cv=Logreg_cv,  # 5-fold 교차 검증
		penalty='l1',
		solver= 'saga',#'saga',
		scoring='accuracy',
		max_iter=1000 ,
		class_weight='balanced'
	)
	Logistic_model.fit(train_norm, train_label_fin)
	train_score = Logistic_model.score(train_norm, train_label_fin)

	best_C = Logistic_model.C_[0]
	print(f"Best C value: {best_C}")
	print("Logistic score: ", train_score)

	# ---- 2-2) Feature selection from  LR efficients
	coeff_used_logistic = np.sum(Logistic_model.coef_ != 0)  # returns the number of coefficients used
	coeffs_logistic = Logistic_model.coef_[0]
	coeffs_rank_logistic = np.where(coeffs_logistic != 0)[0]
	select_rank_logistic = coeffs_rank_logistic[:coeff_used_logistic]

	print("\nLogistic rank: ", select_rank_logistic, len(select_rank_logistic))

	select_feature_train = np.array(train_norm[:, select_rank_logistic], dtype=np.float64)
	select_feature_test = np.array(test_norm[:, select_rank_logistic], dtype=np.float64)

	select_feature_data = []
	select_feature_data.append(select_rank_logistic)
	select_feature_data.append(coeffs_logistic[select_rank_logistic])

	coef_list.append(np.abs(Logistic_model.coef_[0]))

	# ---------- 3. Classifier
	# ---- 2-1) SVM Classifier

	classifier = svm.SVC(C=svm_Cval,gamma=svm_Gval,  probability=True, decision_function_shape='ovo',class_weight='balanced')
	classifier.fit(select_feature_train, np_train_label)
	train_score = classifier.score(select_feature_train, np_train_label)
	test_score = classifier.score(select_feature_test, np_test_label)
	print("[SVM - mean acc.] train score: {0} , test score: {1}".format(train_score, test_score))

	# ---- 2-2) Feature importance based on SVM classifier
	# # SVM에서 중요하게 사용된 특징 추출 // only working at kernel = lenear
	# svm_coef = np.abs(classifier.coef_).mean(axis=0)
	# sorted_indices = np.argsort(svm_coef)[::-1]
	#
	# # lr_features_name = features.columns[coeff_used_logistic]
	# # lr_feature_importance = coeffs_logistic[select_rank_logistic]
	# svm_feature_name = 	lr_features_name[sorted_indices]
	# svm_feature_importance = svm_coef[sorted_indices]
	#
	#
	# print("SVM feature importance:", svm_feature_importance)
	#
	# plt.figure(figsize=(10, 6))
	# sns.barplot(x=svm_feature_importance, y=svm_feature_name)
	# plt.title('Feature Importance based on SVM Coefficients')
	# plt.xlabel('Coefficient Value')
	# plt.ylabel('Feature')
	# plt.show()



	# ---------- 4. Results
	test_cm, test_roc_auc, test_f1score, test_sensitivity, test_specificity, test_correct, test_fpr, test_tpr, test_thr1, test_prediction1 = get_clf_result(
		classifier, select_feature_test, np_test_label)

	train_cm, train_roc_auc, train_f1score, train_sensitivity, train_specificity, train_correct, train_fpr, train_tpr, train_thr1, train_prediction1 = get_clf_result(
		classifier, select_feature_train, np_train_label)

	avg_acc['train'].append(train_score)
	avg_acc['test'].append(test_score)
	avg_f1score['train'].append(train_f1score)
	avg_f1score['test'].append(test_f1score)


	fpr_list.append(test_fpr)
	tpr_list.append(test_tpr)
	auc_list.append(test_roc_auc)

	if test_roc_auc > max_auc:
		max_auc = test_roc_auc
		max_fpr = test_fpr
		max_tpr = test_tpr
		max_index = fold_cnt
		max_clf = classifier
		max_cm = test_cm

		max_selected_test_data = select_feature_test
		max_selected_test_label = np_test_label

	if test_roc_auc < min_auc:
		min_auc = test_roc_auc
		min_fpr = test_fpr
		min_tpr = test_tpr
		min_index = fold_cnt

	if save_result:
		records = []
		records.append(['fold_no', ' Select_No', 'norm_method',
		                 ' SVM_train', ' SVM_test', 'Sensitivity', 'Specificity', 'F1score', 'AUC'])
		records.append([fold_cnt,  coeff_used_logistic, norm_method,
		                train_score, test_score, test_specificity, test_sensitivity, test_f1score,
		                test_roc_auc])

		save_data_2csv(save_path + "/fold_result/svm_result_" + norm_method + str("_ft_num_") + str(coeff_used_logistic) + '.csv',
		               records)
		save_data_2csv(
			save_path + "/fold_result/Selected_features_fold_" + str(fold_cnt).zfill(3) + norm_method + '.csv',
			select_feature_data)


	# --- go to next fold∂
	fold_cnt += 1

auc_list = np.array(auc_list, dtype=np.float64)
avg_auc = np.mean(auc_list)
print("average AUC: ", avg_auc)
print("[average acc] train: {0} \t test: {1}".format(np.mean(avg_acc['train']), np.mean(avg_acc['test'])))
print("[acc std] train: {0} \t test: {1}".format(np.std(avg_acc['train']), np.std(avg_acc['test'])))
print("[average F1-score] train: {0} \t test: {1}".format(np.mean(avg_f1score['train']), np.mean(avg_f1score['test'])))
print("[F1-score std] train: {0} \t test: {1}".format(np.std(avg_f1score['train']), np.std(avg_f1score['test'])))

median_auc = np.nanmedian(auc_list)
# Should be odd number to get exact median value index, ot it will return average of two middle values.
median_index = np.where(auc_list == median_auc)[0][0]
median_fpr = fpr_list[median_index]
median_tpr = tpr_list[median_index]

# ---------- 5. Visualization
# ---- 5-1) AUCROC curve
matplotlib.rcParams.update({'font.size': 12})
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')

plt.plot(max_fpr, max_tpr, '-b', label='best fold (area = {:.3f})'.format(max_auc))
plt.plot(min_fpr, min_tpr, '-g', label='worst fold (area = {:.3f})'.format(min_auc))
plt.plot(median_fpr, median_tpr, '-r', label='Median fold (area = {:.3f})'.format(median_auc))
plt.xlabel('False positive rate', fontsize=14)
plt.ylabel('True positive rate', fontsize=14)
plt.title('ROC curve', fontsize=14)
plt.legend(loc='best')
if save_result:
	plt.savefig(
		save_path + "/meanCW_AUC_ROC_clf_{0}_ftNum{1}_norm_{2}_StratifiedKFold_{3}".format("svm", coeff_used_logistic, norm_method,max_index))
	plt.show()

# ---- 5-2) Confusion Matrix for best fold
disp = ConfusionMatrixDisplay(confusion_matrix=max_cm,display_labels=max_clf.classes_).plot()
if save_result:
	plt.savefig(save_path + "/meanCW_CM_clf_{0}_ftNum{1}_norm_{2}_StratifiedKFold_{3}".format("svm", coeff_used_logistic, norm_method, max_index))
	plt.show()


# # ---- 5-3)  Feature importance based on LR coeff.
# coefficients_mean = np.mean(coef_list, axis=0)
# feature_importance = np.abs(coefficients_mean)
#
#
# importance_df = pd.DataFrame({
# 	'Feature': features.columns,
# 	'Importance': feature_importance
# }).sort_values(by='Importance', ascending=False)
#
# plt.figure(figsize=(10, 6))
# sns.barplot(x='Importance', y='Feature', data=importance_df)
# plt.title('Feature Importance based on Logistic Regression Coefficients')
# plt.show()




