# -*- coding: utf-8 -*-   [ iris data from UCI datasets /  gini vs entropy ---V3 ]  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification
from sklearn.utils.multiclass import unique_labels




#資料下載
df_car = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data', header=None)

#資料預處理
df_car.columns = ["buying","maint", "doors", "persons", "lug_boot", "safety","values"]

target_names = ['unacc','acc', 'good', 'vgood' ]

tabel = {'buying' : {'low' : 0,'med' : 1,'high' : 2,'vhigh' : 3},
         'maint' : {'low' : 0,'med' : 1,'high' : 2,'vhigh' : 3},
         'doors' : { '0' : 0, '1' : 1, '2' : 2, '3' : 3, '4' :4 , '5more' : 5},
         'persons' : {  '2' : 0, '4' : 1 , 'more' : 2},
         'lug_boot' : {  'small' : 0 , 'med' :1 , 'big' : 2},
         'safety' : {  'low' : 0 , 'med' :1 , 'high' : 2},
         'values' : {'unacc' : 0,'acc' : 1, 'good' : 2, 'vgood' : 3}
        }

for index in range(len(df_car['buying'])):
    for columns in df_car.columns.values:
       df_car.ix[index,columns] = tabel[columns][df_car[columns].loc[index]] 



#切割資料
x, y = df_car.iloc[:, 0:6].values, df_car.iloc[:, 6].values
x_train, x_test, y_train, y_test =    train_test_split(x, y, 
                     test_size=0.3, 
                     random_state=0, 
                     stratify=y)


print('-------------------------------------------------------------------------------\n')
print("the size of X_train is : ", len(x_train))
print("the size of X_test  is : ", len(x_test))
print("the size of Y_train is : ", len(y_train))
print("the size of Y_test  is : ", len(y_test)) 

print('-------------------------------------------------------------------------------\n')
#產生gini分類器
clf_gini = tree.DecisionTreeClassifier(criterion = "gini", random_state = 100)

#訓練gini模型
print("gini   =  ", clf_gini)
clf_gini.fit(x_train, y_train)

#產生entropy分類器
print("\n")
clf_entropy = tree.DecisionTreeClassifier(criterion = "entropy", random_state = 100)

#訓練entropy模型
print("entropy  = ", clf_entropy)
clf_entropy.fit(x_train, y_train)

print('-------------------------------------------------------------------------------\n')
#預測
y_pred = clf_gini.predict(x_test)
y_pred_en = clf_entropy.predict(x_test)

#產生混淆陣列
def makeConfusionmatrix(y_true,y_pred):
	df = pd.DataFrame(np.zeros((4,4),dtype=np.int))
	for i in range(len(y_pred)):
		df.ix[y_true[i],y_pred[i]] += 1
	return df

print("y_predication for gini is ", y_pred)
print("Gini Confusion Matrix :")
print(makeConfusionmatrix(y_test,y_pred))

print("\n")

print("y_predication for entropy is ", y_pred_en)
print("entropy Confusion_Matrix :")
print(makeConfusionmatrix(y_test,y_pred_en))

print('-------------------------------------------------------------------------------\n')

#結果評估  自製 precision_recall_fscore_support函數
def resultShow(y_true,y_pred):
	labels = unique_labels(y_true, y_pred)
	tp_sum = np.zeros(len(labels))
	true_sum = np.zeros(len(labels))
	pred_sum = np.zeros(len(labels))

	for i in range(len(y_true)):
		if y_true[i] ==  y_pred[i]:
			tp_sum[y_true[i]] += 1
		true_sum[y_true[i]] += 1
		pred_sum[y_pred[i]] += 1

	precision = tp_sum/pred_sum
	recall = tp_sum/true_sum
	fscore = (2*precision*recall)/(precision+recall)
	support = true_sum.astype('int')
	accuracy = np.sum(tp_sum)/np.sum(pred_sum)

	df = pd.DataFrame({'precision':precision , "recall" : recall,'fscore': fscore,'support':support,'values':target_names})
	df = df.set_index('values')
	print(df)
	print('accuracy : {}'.format(accuracy))


print("Gini Classification report is : ")
resultShow(y_test,y_pred)
print('-------------------------------------------------------------------------------\n')
print("Entropy Classification report is : ")
resultShow(y_test,y_pred_en)


"""
print("Gini Classification report is : ")
#print(classification.classification_report(y_test, y_pred,target_names=target_names))
gini_precision,gini_recall,gini_fscore,gini_support=classification.precision_recall_fscore_support(y_test, y_pred)

df_gini = pd.DataFrame({'gini_precision':gini_precision , "gini_recall" : gini_recall,'gini_fscore': gini_fscore,'gini_support':gini_support,'values':target_names})
df_gini = df_gini.set_index('values')
print(df_gini)


print("Entropy Classification report is : ")
#print(classification.classification_report(y_test, y_pred_en ,target_names=target_names))
entropy_precision,entropy_recall,entropy_fscore,entropy_support=classification.precision_recall_fscore_support(y_test, y_pred_en)

df_entropy = pd.DataFrame({'entropy_precision':entropy_precision , "entropy_recall" : entropy_recall,'entropy_fscore': entropy_fscore,'entropy_support':entropy_support,'values':target_names})
df_entropy = df_entropy.set_index('values')
print(df_entropy)
"""