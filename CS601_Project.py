#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score


# In[2]:


data = pd.read_csv('liver_cirrhosis.csv')

data.head


# In[3]:


#converting all categorical features to be numeric

#status feature conversion

for i in range(data['Status'].shape[0]):
    if data['Status'].iloc[i] == 'C':
        data['Status'].iloc[i] = 0
        
    elif data['Status'].iloc[i] == 'D':
        data['Status'].iloc[i] = 1
        
    elif data['Status'].iloc[i] == 'CL': 
        data['Status'].iloc[i] = 2
    
#drug feature conversion

for i in range(data['Drug'].shape[0]):
    if data['Drug'].iloc[i] == 'Placebo':
        data['Drug'].iloc[i] = 0
    
    elif data['Drug'].iloc[i] == 'D-penicillamine':
        data['Drug'].iloc[i] = 1
    
#sex feature conversion

for i in range(data['Sex'].shape[0]):
    if data['Sex'].iloc[i] == 'M':
        data['Sex'].iloc[i] = 0
        
    elif data['Sex'].iloc[i] == 'F':
        data['Sex'].iloc[i] = 1

#ascites feature conversion

for i in range(data['Ascites'].shape[0]):
    if data['Ascites'].iloc[i] == 'N':
        data['Ascites'].iloc[i] = 0
        
    elif data['Ascites'].iloc[i] == 'Y':
        data['Ascites'].iloc[i] = 1
        
#hepatomegaly feature conversion

for i in range(data['Hepatomegaly'].shape[0]):
    if data['Hepatomegaly'].iloc[i] == 'N':
        data['Hepatomegaly'].iloc[i] = 0
        
    elif data['Hepatomegaly'].iloc[i] == 'Y':
        data['Hepatomegaly'].iloc[i] = 1
        
#spiders feature conversion

for i in range(data['Spiders'].shape[0]):
    if data['Spiders'].iloc[i] == 'N':
        data['Spiders'].iloc[i] = 0
        
    elif data['Spiders'].iloc[i] == 'Y':
        data['Spiders'].iloc[i] = 1
        
#edema feature conversion

for i in range(data['Edema'].shape[0]):
    if data['Edema'].iloc[i] == 'N':
        data['Edema'].iloc[i] = 0
        
    elif data['Edema'].iloc[i] == 'Y':
        data['Edema'].iloc[i] = 1
    
    elif data['Edema'].iloc[i] == 'S':
        data['Edema'].iloc[i] = 2


# In[4]:


data.head


# In[5]:


X = data[['N_Days', 'Status', 'Drug', 'Age', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema', 'Bilirubin', 'Cholesterol', 'Albumin', 'Copper','Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin']]

y = data['Stage']


# In[41]:


y.head


# In[7]:


#ensuring no null values are present in the dataset

tbd_index = []

for i in range(data.shape[0]):
    for item in list(data.iloc[i]):
        if str(item) == 'nan':
            tbd_index.append(i) 
            break
tbd_index.reverse()


# In[8]:


print('Samples containing null values:', len(tbd_index))


# In[46]:


def bootstrap(X, y, n_samples):
    new_X = []
    new_y = []
    for i in range(n_samples):
        rand_index = random.randint(0, (n_samples - 1))
        new_X.append(X.iloc[rand_index])
        new_y.append(y.iloc[rand_index])
    return pd.DataFrame(new_X), pd.DataFrame(new_y)

def find_splits(col):
    midpoints = []
    sorted_col = sorted(col)
    for i in range(0, len(col) - 1):
        midpoints.append((sorted_col[i] + sorted_col[i + 1]) / 2)
    return list(set(midpoints))

def split_data(feature, reg_target, split_point):
    mask = feature >= split_point
    anti_mask = feature < split_point
    x_true = feature[mask]
    y_true = reg_target[mask]
    x_false = feature[anti_mask]
    y_false = reg_target[anti_mask]
    return x_true, y_true, x_false, y_false

def MAE(y, y_true, y_false):
    y_true_hat = np.mean(y_true)
    y_false_hat = np.mean(y_false)
    y_hat = np.mean(y)
    root_mae = np.mean(np.abs(y - y_hat))
    weighted_mae = len(y_true)/len(y) * np.mean(np.abs(y_true - y_true_hat)) + len(y_false)/len(y) * np.mean(np.abs(y_false - y_false_hat))
    return root_mae, weighted_mae

def get_best_split_of_features(data, feature_index):
    data = data.to_numpy()
    n_features = data.shape[1] - 1
    list_errors = {}
    min_err = 10000000
    target_index = 0
    
    for feature in feature_index:
        x = data[:, feature]
        y = data[:, target_index]
        split_points = find_splits(x)
        for point in split_points:
            x_true, y_true, x_false, y_false = split_data(x, y , point)
            root_mae, weighted_mae = MAE(y, y_true, y_false)
            list_errors[feature, point] = (weighted_mae, root_mae)
        
            if weighted_mae < min_err:
                min_err = weighted_mae
                best_feature = feature
                best_point = point
    return list_errors, best_feature, best_point

def dt_with_random_feature_selection(data, sample):
    
    #randomly select m features while ensuring at least one feature is selected
    
    feature_index = []
    for i in range(1, 19):
        if random.randint(0, 1) == 1:
            feature_index.append(i)
    if len(feature_index) == 0:
        feature_index.append(random.randint(1,19));
    
    list_err, best_feature, best_point = get_best_split_of_features(data, feature_index)
    if sample[best_feature] >= best_point: 
        df = data[data.iloc[:, best_feature] >= best_point].iloc[:, 0].mean()
        return max(set(list(df)), key=list.count)
    else:
        df = data[data.iloc[:, best_feature] <  best_point].iloc[:, 0].mean()
        return max(set(list(df)), key=list.count)

#Part C: allow user to decide number of random features selected

def dt_with_set_random_features(data, y, sample, num_features):
    
    feature_index = []
    while len(feature_index) < num_features:
        new_index = random.randint(1, 18)
        if new_index not in feature_index:
            feature_index.append(new_index)
    
    list_err, best_feature, best_point = get_best_split_of_features(data, feature_index)
    if sample[best_feature] >= best_point: 
        df = y.iloc[[data.iloc[:, best_feature] >= best_point]]
        return max(set(list(df)), key=List.count)
    else:
        df = y.iloc[[data.iloc[:, best_feature] <  best_point]]
        return max(set(list(df)), key=List.count)
    
def random_forest_classifier_from_scratch(X, y, sample, n_trees, max_features, n_samples):
    tree_oob_and_prediction = {}
    for i in range(n_trees):
        bs_X, bs_y = bootstrap(X, y, n_samples)
        oob_indices = [x for x in list(X.index) if x not in bs_X.index]
        full_bootstrap = bs_y.join(bs_X)
        prediction = dt_with_set_random_features(full_bootstrap, bs_y, sample, max_features)
        tree_oob_and_prediction['tree' + str(i)] = oob_indices, prediction
    return tree_oob_and_prediction
        
        
# part c: average the predictions from all trees for a given input

def aggregate_prediction(tree_oob_and_prediction):
    li = []
    for (li, val2) in tree_oob_and_prediction.values():
        li.append(val2)
    print(li)
    return max(set(li), key=list.count)
        


# In[45]:


y.iloc[[5, 8]]


# In[47]:


random_sample = data.iloc[4250, :]

res = random_forest_classifier_from_scratch(X, y, random_sample, 20, 18, 20000)


# In[29]:


aggregate_prediction(res)


# In[48]:


#train random forest classifier using sklearn to compare results

X_train,X_test,y_train,y_test = train_test_split (X , y, test_size = 0.4, random_state = 47, stratify = y)

rf_model = RandomForestClassifier(n_estimators=200, random_state=47)
rf_model.fit(X_train,y_train)

y_preds = rf_model.predict(X_test)

cm = confusion_matrix(y_test,y_preds)


# In[49]:


cm


# In[50]:


cm = np.flip(cm)


# In[51]:


cm
       


# In[52]:


cm.ravel().sum()


# In[53]:


#calculations for stage 1

FP_stage1 = cm[1][0] + cm[2][0]
FN_stage1 = cm[0][1] + cm[0][2]
TP_stage1 = cm[0][0]
TN_stage1 = cm[1][1] + cm[1][2] + cm[2][1] +cm[2][2]

prec_stage1 = TP_stage1 / (TP_stage1 + FP_stage1)
recall_stage1 = TP_stage1 / (TP_stage1 + FP_stage1)
f1_stage1 = 2 * (prec_stage1* recall_stage1) / (prec_stage1 + recall_stage1)

#calculations for stage 2

FP_stage2 = cm[0][1] + cm[2][1]
FN_stage2 = cm[1][0] + cm[1][2]
TP_stage2 = cm[1][1]
TN_stage2 = cm[0][0] + cm[0][2] +cm[2][0] + cm[2][2]

prec_stage2 = TP_stage2 / (TP_stage2 + FP_stage2)
recall_stage2 = TP_stage2 / (TP_stage2 + FP_stage2)
f1_stage2 = 2 * (prec_stage2* recall_stage2) / (prec_stage2 + recall_stage2)

#calculation for stage 3

FP_stage3 = cm[0][2] + cm[1][2]
FN_stage3 = cm[2][0] + cm[2][1]
TP_stage3 = cm[2][2]
TN_stage3 = cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1]

prec_stage3 = TP_stage3 / (TP_stage3 + FP_stage3)
recall_stage3 = TP_stage3 / (TP_stage3 + FP_stage3)
f1_stage3 = 2 * (prec_stage3 * recall_stage3) / (prec_stage3 + recall_stage3)


# In[54]:


TP_stage1 + TP_stage2 + TP_stage3


# In[55]:


recall_stage3


# In[57]:


#final metric calculations

accuracy_sklearn =  ((TP_stage1 + TP_stage2 + TP_stage3) / cm.ravel().sum()) * 100

print('Accuracy with SKLearn Implementation:', accuracy_sklearn)

precision_sklearn = (prec_stage1 + prec_stage2 + prec_stage3) / 3 * 100

print('Precision with SKLearn Implementation:', precision_sklearn)

recall_sklearn = (recall_stage1 + recall_stage2 + recall_stage3) / 3 * 100

print('Recall with SKLearn Implementation:', recall_sklearn)

f1_score_sklearn = (f1_stage1 + f1_stage2 + f1_stage3) / 3 * 100

print('F1-Score with SKLearn Implementation:', f1_score_sklearn)

