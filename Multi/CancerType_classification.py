from sklearn import svm, metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from Utils import save_data_2csv
from sklearn.metrics import  accuracy_score,classification_report

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold

import warnings
warnings.filterwarnings("ignore")

no_seed =  25 #
np.random.seed(no_seed)  # HK added this line
split_seed = np.random.randint(10000, size=2000)  # HK_added this line: 1000 = iteration number


# ---------  0. Set parameters
norm_method = "z-score"  # ['z-score', 'min-max', 'none', 'quantile']
select_clf = 'svm'
save_result = True
test_acc_list = []
clf_cutoff =0.5

Cval = 50
Gval = 1.2

feature_name = [' DCD', ' GFRA1', ' Her2', ' IGFR1', ' SMR3B', ' CEA',
                ' GPA33', ' MEP1A', ' SLC12A2', ' CD63', ' CD81', ' CD9',
                ' ASGR1', ' FGB', ' RBP4', ' SELENOP', ' c16orf89', ' DSG3',
                ' SCGB1A1', ' SP-B', ' Amy2A', ' GP2', ' REG1b', ' WNT2',
                ' CD24', ' EGFR', 'EpCAM', ' PDL-1', ' EGFRvIII', ' PDGFRalpha']
# , ' Age', ' Gender']


min_acc = 1
max_acc = 0

min_index = 0
max_index = 0


max_clf = None
max_selected_test_data = None
max_selected_test_label = None

max_selected_train_data = None
max_selected_train_label = None
max_test_class_probs = []
max_train_class_probs = []
max_thr = []


# ---------  1. Data load
cancer_types = ['Br', 'Cl', 'Li', 'Lu', 'Pa']
# # Br is 0, Cl is 1, Li is 2, Lu is 3, Pa is 4

# meta_data file contains: tumor type, Dx label, Patient ID
data_file = "Data/features_median_ratio90_pp0406_v2_30_multi_exceptPA9.npy"
meta_file = 'Data/data_file_pp0406_v2_30_multi_exceptPA9.csv'

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

save_folder =  'Result/5types_cancer/Alltrain/LR_SVM_C{0}G{1}/'.format(Cval, Gval)

save_path_predict = save_folder + select_clf + '_predict/'
if not os.path.exists(save_path_predict):
    os.makedirs(save_path_predict)

save_path_label = save_folder + select_clf + '_label/'
if not os.path.exists(save_path_label):
    os.makedirs(save_path_label)


save_path = save_folder
if not os.path.exists(save_path):
	os.makedirs(save_path)


if save_result:
	if not os.path.exists(save_path + "/dataset/"):
		os.makedirs(save_path + "/dataset/")

	# data_train.to_csv(save_path + "/dataset/{0}_train_data.csv".format(iteration), index_label='sample_ID')
	# data_valid.to_csv(save_path + "/dataset/{0}_valid_data.csv".format(iteration), index_label='sample_ID')
	# data_test.to_csv(save_path + "/dataset/{0}_test_data.csv".format(iteration), index_label='sample_ID')
	#
	# label_train.to_csv(save_path + "/dataset/{0}_train_label.csv".format(iteration), index_label='sample_ID')
	# label_valid.to_csv(save_path + "/dataset/1_valid_label.csv".format(iteration), index_label='sample_ID')
	# label_test.to_csv(save_path + "/dataset/{0}_test_label.csv".format(iteration), index_label='sample_ID')

	# # 0-3. final result (size)
	#print(" --------- data size & components --------")
	#print("------[Entire dataset] -----\n total size: {0}  \n each label :\n{1}".format(labels.size, labels.value_counts()))
	# print("----- [train] -----\n size: {0}  \n each label :\n{1}".format(label_train.size, label_train.value_counts()))
	# print("----- [valid] -----\n size: {0}  \n each label :\n{1}".format(label_valid.size, label_valid.value_counts()))
	# print("----- [test] -----\n size: {0}  \n each label :\n{1}".format(label_test.size, label_test.value_counts()))


# ---------  1. Normalization
if norm_method == 'z-score':
	scalar = StandardScaler()
	df_scaled = pd.DataFrame(scalar.fit_transform(features), columns = features.columns)
	train_norm = np.array(df_scaled)
elif norm_method == 'min-max':
	scalar = MinMaxScaler(feature_range=(0, 10))
	scalar.fit(features)
	train_norm = scalar.transform(features)
elif norm_method == 'quantile':
	scaler = QuantileTransformer()
	scaler.fit(features)
	train_norm = scaler.transform(features)
elif norm_method == 'none':
	train_norm = np.array(features)

np_train_label = np.array(labels)

# ---------  2. Feature selection
# ---- 2-1) Logistic regression
Logreg_cs = [0.01, 0.1, 0.5, 0.8, 1, 1.2, 2, 5, 10, 30,100]
Logreg_cv = StratifiedKFold(n_splits=5, shuffle=False)

Logistic_model = LogisticRegressionCV(
	Cs=Logreg_cs,  # 10개의 다른 C 값 테스
	cv=Logreg_cv,  # 5-fold 교차 검증
	penalty='l1',
	solver= 'saga',#'saga',
	scoring='accuracy',
	max_iter=1000 ,
	class_weight='balanced',
	multi_class='multinomial'
)
Logistic_model.fit(train_norm, labels)
train_score = Logistic_model.score(train_norm, labels)
print("Optimal C: ", Logistic_model.C_[0])
print("Logistic score: ", train_score)


# Coefficients for each class
coefficients = Logistic_model.coef_
mean_coefficients = np.mean(np.abs(coefficients), axis=0)


num_top_features = 21 #np.sum(Logistic_model.coef_ != 0) # 21
top_features_idx = np.argsort(mean_coefficients)[-num_top_features:]
select_rank_logistic = top_features_idx
select_feature_train = np.array(train_norm[:, select_rank_logistic], dtype=np.float64)

select_feature_data = []
select_feature_data.append(select_rank_logistic)
select_feature_data.append(mean_coefficients[select_rank_logistic])


# ---- 2-2) Feature importance
feature_importance = np.abs(Logistic_model.coef_[0])
class_coeff = Logistic_model.coef_
selected_features = train_norm[:, select_rank_logistic]


# ---------  3. Classifier
# ---- 3-1) SVM classifier
classifier = svm.SVC(C=Cval, gamma=Gval, kernel='linear', probability=True, decision_function_shape='ovo', class_weight='balanced')

classifier.fit(select_feature_train, labels)
train_y_pred = classifier.predict(select_feature_train)

train_clf_score = classifier.score(select_feature_train, np_train_label)
thr_train_acc = accuracy_score(np_train_label, train_y_pred)
test_acc_list.append(thr_train_acc)

print("[clf score] train: {0}\t test: {1}".format(train_clf_score, thr_train_acc))


# ---- 3-2) Feature importance based on SVM coeff.
svm_coef = np.abs(classifier.coef_).mean(axis=0)
svm_feature_importance = svm_coef[np.argsort(svm_coef)[::-1]]

feature_importance = svm_coef[np.argsort(svm_coef)[::-1]]
feature_names =  features.columns[top_features_idx]

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance, y=feature_names)
plt.title('Feature Importance based on SVM Coefficients')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.show()

FS_save_df= pd.DataFrame([feature_importance, feature_names])
FS_save_df.to_csv(save_path + "/SVM_Feature_importance.csv")

# --------- 4. Result
clf_predict = classifier.predict_proba(select_feature_train)
predcs1 = clf_predict[:, 1]

iteration=0
predict_save = save_path_predict + str(iteration).zfill(3) + '_.npy'
np.save(predict_save, clf_predict)

label_save = save_path_label + str(iteration).zfill(3) + '_.npy'
np.save(label_save, train_y_pred)
prediction = np.argmax(clf_predict, axis=1)



np_train_label_flat = np_train_label.flatten()
train_y_pred_flat = train_y_pred.flatten()

test_cm = metrics.confusion_matrix(np_train_label_flat, train_y_pred_flat)
correct = (prediction == np_train_label).mean()
f1score = metrics.f1_score(np_train_label, prediction, average='macro')
sensitivity = test_cm[0, 0] / (test_cm[0, 0] + test_cm[0, 1])
specificity = test_cm[1, 1] / (test_cm[1, 0] + test_cm[1, 1])



# ---- 4-2) Confusion matrix and report
from sklearn.metrics import ConfusionMatrixDisplay

disp = ConfusionMatrixDisplay(confusion_matrix=test_cm,display_labels=cancer_types).plot()
if save_result:
	plt.savefig(save_path + "/meanCW_CM_clf_{0}_ftNum_norm_{2}_StratifiedKFold_{3}".format("svm", num_top_features, norm_method, max_index))
	plt.show()


if not os.path.exists(save_path + "/cm_report/"):
	os.makedirs(save_path + "/cm_report/")
train_cm_report = classification_report(np_train_label, train_y_pred, output_dict= True)
train_cm_df = pd.DataFrame(train_cm_report).transpose()
train_cm_df.to_csv(save_path + "/cm_report/cm_report_5types_ovo_iteration_{0}.csv".format(iteration), index=True)


# ---- 4-2) Accuracy
acc_list = np.array(test_acc_list, dtype=np.float64)
print(" ------ [Accuracy Summary] ------ \n\t Max {0} \t Min Acc: {1} \t Avg. Acc: {2} ".format(np.max(acc_list), np.min(acc_list), np.mean(acc_list)))
print(max_index)


with open(save_path +'/5types_ovo_best_model.pickle','wb') as fw:
    pickle.dump(max_clf, fw)

if not os.path.exists(save_path + "records/"):
	os.makedirs(save_path + "records/")

# print(prediction)
save_data_2csv(save_path + 'records/train_selected_features_best_model.csv',select_feature_train)
save_data_2csv(save_path + 'records/train_label_best_model.csv',np_train_label)
save_data_2csv(save_path + 'records/train_predictions.csv',clf_predict) #clf_predict)

