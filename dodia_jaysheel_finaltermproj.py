#!/usr/bin/env python
# coding: utf-8

# In[73]:


from IPython.display import Image


# # CS634 Fall Final Project 

# Name: Jaysheel Dodia<br>
# UCID: jd849<br>
# MailID: jd849@njit.edu<br>
# Date: 24/11/2024 (DD/MM/YYYY)<br>
# Instructur: Dr. Yasser Abduallah<br>
# Class: CS634-101<br>

# # Abstract

# In this project we will perform classification of emails into "spam" and "ham" categories using Machine Learning and Deep Learning, specifically using Random Forest Classification, Support Vector Machines (SVM) and Long Short-Term Memory (LSTM) neural networks. The goal is to develop a model which effectively classifies the email as a "spam" or a "ham" using Random Forest, SVM and LSTM. This project uses publicly available Spam Classification Dataset. We implement data preprocessing techniques, and evaluates the models' performance using metrics such as accuracy, precision, recall, and F1-score. We also perform K-Fold validation on the dataset on each of the 3 algorithms and compare the resutls using the evaluation metrics. 

# # Introduction

# In this project applies multiple Machine Learning and Deep Learning algorithms - Random Forest Classification, Support Vector Machine and Long Short-Term Memory (LSTM) network - to classify emails as "spam" or "ham". The project involves preprocessing a publicly available dataset, training multiple models using K-Fold Cross Validation, and evaluating performance using metrics. By doing this we are able to compare the performance of different models and their effectiveness for training a text-classification model.

# # Methods

# ## Random Forest Classification

# Random Forest is an ensemble learning algorithm works by creating multiple decision trees during training and outputs the mode of the classes (classification) or mean prediction (regression) of the individual trees. It is an extension of the Decision Tree algorithm and is widely used due to its simplicity, flexibility, and robustness. Random Forest overcomes many of the limitations of a single decision tree by combining the predictions of multiple trees. This reduces the risk of overfitting and improving predictive performance.

# ## Support Vector Machine (SVM)

# A support vector machine (SVM) is a supervised machine learning algorithm that classifies data by finding an optimal line or hyperplane that maximizes the distance between each class in an N-dimensional space. It works by diving the virtual plane into maximum separating hyperplane between the different classes in the target feature, making them suitable for classification tasks.

# ## Long Short-Term Memory (LSTM)

# Long Short-Term Memory is an improved version of recurrent neural network (RNN). Learning long-term dependencies might be challenging for a typical RNN because it only has one hidden state that is transferred through time. The memory cell, a container that can store information for a long time, is introduced in the LSTM model to solve this issue.

# ## Dataset

# The Spam Ham text classification dataset is a widely used collection of text messages or emails labeled as either spam (unwanted/harmful) or ham (legitimate). This dataset serves as a valuable resource for developing and evaluating machine learning and deep learning models in natural language processing tasks, particularly for spam detection and text classification.
# The dataset can be downloaded from the following link: [Dataset Link](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

# ## Libraries Used

# |Library|Usage| Version |
# | -------- | -------- |-------- |
# |Numpy | For matrix operations| 2.0.2 |
# |Pandas | Data manipulations and analysis| 2.2.3 |
# |Matplotlib | Graph Plotting library| 3.9.2 |
# |Seaborn | High-level plotting based on Matplotlib| 0.13.2 |
# |Scikit-Learn | Machine Learning and Data Pre-processing| 1.5.2 |
# |Tensorflow | Build Neural Network and LSTM| 2.18.0 |

# ## Prerequisite

# 1. Open up a terminal in our project directory (Refer Screenshot 1)
# 2. Now create a python virtual environment using the code `python -m venv venv` (Refer screenshot 2)
# 3. Now we will activate the environment to use it using the command `.\venv\Scripts\activate` if you're on windows. If you're on linux activate the environment using the command `source venv/bin/activate` (Refer Screenshot 3. I am using windows).
# 4. Once activated, install the requirements for our project using the command `pip install -r requirements.txt`. (Refer Screenshot 4)
# 5. Now, we can execute our python file by writing `python dodia_jaysheel_finaltermproj.py`. (Refer Screenshot 5)

# ## Note
# * The graph plots used for visualization may be displayed differently while running the python file compared to the jupyter notebook. This is due to a change in the visual environment.<br>
# * On running the python file, when the graphs are generated, you will have to close the graph's window for the script to execute further. This happens because the `plt.show()` function blocks the terminal execution as long as the graph is being displayed. Close the graph window to continue.

# #### Screenshot 1

# In[74]:


Image('ss/1.png')


# #### Screenshot 2

# In[75]:


Image('ss/2.png')


# #### Screenshot 3

# In[76]:


Image('ss/3.png')


# #### Screenshot 4

# In[77]:


Image('ss/4.png')


# #### Screenshot 5

# In[78]:


Image('ss/5.png')


# # Implementation

# ## Import Libraries

# Import all the required libraries to carry out preprocessing tasks, implement Machine Learning algorithms and build LSTM neural network.

# In[79]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, brier_score_loss, roc_curve, roc_auc_score

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding


# In[80]:


# Set random seed
np.random.seed(42)


# ## Load Dataset

# * Loading the dataset to our pandas dataframe
# * The dataset consists of 2 main columns: Messages and Category
# * Message: Contains the data or text of our message
# * Category: Specifies if the message is a spam or a ham

# In[81]:


df = pd.read_csv("./dataset/spam.csv", encoding="latin-1")
df.head()


# In[82]:


df.columns


# ## Preprocess Columns

# In[83]:


cols_to_drop = ["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"]
df = df.drop(cols_to_drop, axis=1)


# In[84]:


df.columns = ["Category", "Message"]


# In[85]:


df.head()


# ## Reducing dataset size

# Random sampling our dataset to extract 2,500 rows at random and reduce the size of our dataset.

# In[86]:


df = df.sample(2500)


# In[87]:


df.shape


# ## Label Encoding

# Converting target variable to 0 and 1 instead of the text `ham` and `spam`. Set the target variable to 1 if the category is "spam" and 0 if the category is "ham"

# In[88]:


df['spam']=df['Category'].apply(lambda x: 1 if x=='spam' else 0)
df.head()


# In[89]:


df.sample(1)


# ## Preprocess - For ML

# ### Count Vectorizer

# Count Vectorizer converts a collection of text documents (a dataframe of rows in our case) to a matrix of token count.

# In[90]:


cv = CountVectorizer()
X_cv = cv.fit_transform(df['Message'])
y = df['spam']


# ## Functions to Calculate Metrics

# ### Function to create the confusion matrix

# In[91]:


def create_cm(tp, tn, fp, fn):
    cm = np.array([[tp, fp], [fn, tn]])
    return cm


# ### Function to calculate Brier Skill Score (BSS)

# In[92]:


def calc_bss(y_test, bs):
    mean = np.mean(y_test)
    brier_ref = np.mean((y_test - mean) ** 2)
    bss = bs/brier_ref
    return bss


# ### Function to calculate all metrics

# In[93]:


def calc_all_metrics(y_test, y_pred, i):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Rate
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * precision * recall) / (precision + recall)
    
    auc_score = roc_auc_score(y_test, y_pred)

    brier = brier_score_loss(y_test, y_pred)
    brier_skill_score = calc_bss(y_test, brier)

    cm = create_cm(tp, tn, fp, fn)
    
    print("Metrics for Fold ", i)
    print("-"*10)
    print("Confusion Matrix for Fold ", i)
    print(cm)
    print("Number of True Positives ", tp)
    print("Number of False Positives ", fp)
    print("Number of True Negatives ", tn)
    print("Number of False Negatives ", fn)
    print("True Positive Rate ", tpr)
    print("True Negative Rate ", tnr)
    print("False Positive Rate ", fpr)
    print("False Negative Rate ", fnr)
    print("Accuracy ", accuracy)
    print("Precision ", precision)
    print("Recall ", recall)
    print("F1 Score ", f1)
    print("AUC Score ", auc_score)
    print("Brier Score ", brier)
    print("Brier Skill Score ", brier_skill_score)
    print(f"Training for fold {i} completed")
    print("\n")
    
    
    fold = f"Fold {i}"
    return [fold, tp, tn, fp, fn, tpr, tnr, fpr, fnr, accuracy, precision, recall, f1, auc_score, brier, brier_skill_score]


# ## Plotting Function

# Function to plot the AUC-ROC Curve for each fold of the model

# In[94]:


def plot_roc_curves_in_grid(roc_data, model_name):
    n_rows = int(np.ceil(len(roc_data) / 3))
    fig, axes = plt.subplots(n_rows, 3, figsize=(10, n_rows * 3))
    axes = axes.flatten()  # Flatten the axes to make indexing easier

    for i, (fpr, tpr, auc_score) in enumerate(roc_data):
        ax = axes[i]
        ax.plot(fpr, tpr, label=f"ROC Curve (AUC) score {auc_score:.2f}")
        ax.plot([0, 1], [0, 1], linestyle='--')  # Diagonal line (random classifier)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{model_name} Fold {i + 1} ROC Curve')
        
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout(pad=3.0)
    plt.show()


# ## Creating 10-Fold

# The dataset is divided into 10 equal parts (called folds), where each fold is used once as a test set while the remaining 9 folds are used as the training set.

# In[95]:


N_SPLITS = 10
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)


# ## Random Forest Classifier

# In[96]:


rf_metrics = []
rf_roc_data = []

print("-"*20)
print("Starting Random Forest Training")
print("-"*20)

# Loop through the 10Fold splits
for i, (train_index, test_index) in enumerate(kf.split(X_cv), start=1):
    # Splitting the data into train and test split
    X_train, X_test = X_cv[train_index], X_cv[test_index] # type: ignore
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    print(f"Training Fold {i}")
    
    # Training the model
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # Calculating the AUC-ROC curve and score
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred)
    rf_roc_data.append((fpr, tpr, auc_score))
    
    # Calculate all the metrics and store them
    rf_metrics.append(calc_all_metrics(y_test, y_pred, i))


# In[97]:


# Add model name to column
for i, metric in enumerate(rf_metrics):
    metric.insert(0, 'Random Forest')


# In[98]:


plot_roc_curves_in_grid(rf_roc_data, "Random Forest")


# ## SVM

# In[99]:


svm_metrics = []
svm_roc_data = []

print("-"*20)
print("Starting SVM Training")
print("-"*20)

# Loop through the KFold splits
for i, (train_index, test_index) in enumerate(kf.split(X_cv), start=1):
    # Splitting the data into train and test split
    X_train, X_test = X_cv[train_index], X_cv[test_index] # type: ignore
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    print(f"Training Fold {i}! ", end='')
    # Training the model
    svm = SVC()
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)

    # Calculating the AUC-ROC curve and score
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred)
    svm_roc_data.append((fpr, tpr, auc_score))
    
    # Calculate metrics and storing them
    svm_metrics.append(calc_all_metrics(y_test, y_pred, i))


# In[100]:


# add the model name to the metrics
for i, metric in enumerate(svm_metrics):
    metric.insert(0, 'SVM')


# In[101]:


plot_roc_curves_in_grid(svm_roc_data, "SVM")


# ## Preprocess for LSTM

# In[102]:


X = df['Message']
y = df['spam']


# Break the text into tokens and make them suitable for training an LSTM.

# In[103]:


# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
X_tok= tokenizer.texts_to_sequences(X)

# Padding the data
X_pad = pad_sequences(X_tok)


# ## LSTM

# ### Function to build LSTM model 

# In[104]:


def build_model(X_train_pad):
    # Training the model
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=32))
    model.add(LSTM(32))
    model.add(Dense(1, activation='sigmoid'))
    return model


# ### Executing

# In[105]:


lstm_metrics = []
lstm_roc_data = []
print("-"*20)
print("Starting LSTM Training")
print("-"*20)
for i, (train_index, test_index) in enumerate(kf.split(X_pad), start=1):
    # Splitting the data into train and test split
    X_train, X_test = X_pad[train_index], X_pad[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # train the LSTM model
    model = build_model(X_train)

    model.compile(loss='binary_crossentropy', optimizer='adam')
    
    model.fit(X_train, y_train, epochs=2, batch_size=32, verbose=0)
    y_pred = model.predict(X_test, verbose=0) > 0.5

    # Calculating the AUC-ROC score and curve
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred)
    lstm_roc_data.append((fpr, tpr, auc_score))

    # calculating the metrics and storing them
    lstm_metrics.append(calc_all_metrics(y_test, y_pred, i))
   


# In[106]:


for i, metric in enumerate(lstm_metrics):
    metric.insert(0, 'LSTM')


# In[107]:


plot_roc_curves_in_grid(lstm_roc_data, "LSTM")


# ## Tabulate

# Store the data into table format using pandas dataframe 

# In[108]:


columns = ['ModelName', 'Fold', 'TP', 'TN', 'FP', 'FN', 'TPR', 'TNR', 
           'FPR', 'FNR', 'Accuracy', 'Precision', 
           'Recall', 'F1', 'AUC', 'Brier', 'Brier Skill Score']
rf_metrics_df = pd.DataFrame(rf_metrics, columns=columns)
svm_metrics_df = pd.DataFrame(svm_metrics, columns=columns)
lstm_metrics_df = pd.DataFrame(lstm_metrics, columns=columns)


# ### Metrics in Table Format

# #### Random Forest Classifier 

# In[109]:


rf_metrics_df


# #### SVM

# In[110]:


svm_metrics_df


# #### LSTM

# In[111]:


lstm_metrics_df


# In[112]:


# combine all the dataframes
all_metrics = pd.concat([rf_metrics_df, svm_metrics_df, lstm_metrics_df], axis=0)
all_metrics.reset_index(drop=True, inplace=True)
all_metrics


# In[113]:


all_metrics.shape


# ## Calculate Mean

# In[114]:


all_metrics.drop('Fold', axis=1).groupby('ModelName').mean()


# ### Print in terminal

# In[115]:


# print in terminal
def print_all_metrics(all_metrics):
    mean_df = all_metrics.drop('Fold', axis=1).groupby('ModelName').mean()
    for col in mean_df.columns:
        print(f"Mean {col} for each model")
        print(mean_df[col])
        print("\n")
print("-"*30)
print("Mean of Metrics for each model")
print("-"*30)
print_all_metrics(all_metrics)
print("-"*30)


# # Visualization

# ## Plot all metrics of Random Forest Classifier for each fold

# In[116]:


rf_metrics_df_plot = pd.DataFrame(rf_metrics, columns=columns)
rf_metrics_df_plot.drop('Fold', axis=1, inplace=True)
# List of metrics to plot
metrics = ['TP', 'TN', 'FP', 'FN', 'TPR', 'TNR', 'FPR', 'FNR', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'Brier', 'Brier Skill Score']

# Plotting the metrics for each fold
fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 40))
fig.suptitle('Random Forest Metrics Across Folds', fontsize=16)

for i, metric in enumerate(metrics):
    axes[i].plot(rf_metrics_df['Fold'], rf_metrics_df[metric], marker='o')
    axes[i].set_title(f'{metric} Across Folds')
    axes[i].set_xlabel('Fold')
    axes[i].set_ylabel(metric)
    axes[i].grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()


# ## Plot all metrics of SVM for each fold

# In[117]:


import matplotlib.pyplot as plt

svm_metrics_df_plot = pd.DataFrame(svm_metrics, columns=columns)
svm_metrics_df_plot.drop('Fold', axis=1, inplace=True)
# List of metrics to plot
metrics = ['TP', 'TN', 'FP', 'FN', 'TPR', 'TNR', 'FPR', 'FNR', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'Brier', 'Brier Skill Score']

# Plotting the metrics for each fold
fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 40))
fig.suptitle('SVM Metrics Across Folds', fontsize=16)

for i, metric in enumerate(metrics):
    axes[i].plot(svm_metrics_df['Fold'], svm_metrics_df[metric], marker='o')
    axes[i].set_title(f'{metric} Across Folds')
    axes[i].set_xlabel('Fold')
    axes[i].set_ylabel(metric)
    axes[i].grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()


# ## Plot all metrics of LSTM for each fold

# In[118]:


lstm_metrics_df_plot = pd.DataFrame(lstm_metrics, columns=columns)
lstm_metrics_df_plot.drop('Fold', axis=1, inplace=True)
# List of metrics to plot
metrics = ['TP', 'TN', 'FP', 'FN', 'TPR', 'TNR', 'FPR', 'FNR', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'Brier', 'Brier Skill Score']

# Plotting the metrics for each fold
fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 40))
fig.suptitle('LSTM Metrics Across Folds', fontsize=16)

for i, metric in enumerate(metrics):
    axes[i].plot(lstm_metrics_df['Fold'], lstm_metrics_df[metric], marker='o')
    axes[i].set_title(f'{metric} Across Folds')
    axes[i].set_xlabel('Fold')
    axes[i].set_ylabel(metric)
    axes[i].grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.97], pad=3.0)
plt.show()


# ## BoxPlot to compare models

# In[146]:


sns.set_theme(style="whitegrid")

# List of metrics to plot
metrics = ['TP', 'TN', 'FP', 'FN', 'TPR', 'TNR', 'FPR', 'FNR', 'Accuracy', 'Precision', 
           'Recall', 'F1', 'AUC', 'Brier', 'Brier Skill Score']

fig, axes = plt.subplots(5, 3, figsize=(18,18))
axes = axes.flatten()

for i, metric in enumerate(metrics):
    sns.boxplot(x='ModelName', y=metric, data=all_metrics, ax=axes[i])
    axes[i].set_title(f'Boxplot of {metric}')

# Adjust the layout
plt.tight_layout(pad=3.0)
plt.show()


# # Observation

# 1. **True Positive (TP)**: LSTM has got more TP for one of the fold than RandomForest and SVM.
# 2. **True Negative (NP)**: All the 3 models have somewhat similar number of TNs.
# 3. **False Positives (FP)**: RF and SVM have lowest count of FP compared to LSTM which shows some variation. We can also observe that there are some outliers in this case for RF and SVM.
# 4. **False Negatives (FN)**: RF and SVM have slightly higher number of FNs than LSTM
# 5. **True Positive Rate (TPR)**: LSTM has a higher TPR compared to RF and SVM, indicating better sensitivity.
# 6. **True Negative Rate (TNR)**: All models have similar TNR, with LSTM showing slightly more variation.
# 7. **False Positive Rate (FPR)**: RF and SVM have lower FPR compared to LSTM, indicating fewer false alarms.
# 8. **False Negative Rate (FNR)**: LSTM has a lower FNR compared to RF and SVM, indicating fewer missed detections.
# 9. **Accuracy**: LSTM shows higher accuracy across folds compared to RF and SVM.
# 10. **Precision**: All the models have similar precision, with LSTM showing slightly higher variation.
# 11. **Recall**: LSTM has higher recall, indicating better sensitivity.
# 12. **F1 Score**: LSTM has a higher F1 score, indicating a better balance between precision and recall.
# 13. **AUC**: LSTM has a higher AUC, indicating better overall performance.
# 14. **Brier Score**: LSTM has a lower Brier score, indicating better probabilistic predictions.
# 15. **Brier Skill Score**: LSTM has a lower Brier Skill Score, indicating better probabilistic predictions.

# # Discussion

# Based on the observations we can say that the LSTM model performs better than the Random Forest and SVM models in a number of important metrics. Overall accuracy, precision, recall, and F1 score are all higher for the LSTM model, along with a reduced False Negative Rate (FNR) and a higher True Positive Rate (TPR). This suggests that the LSTM model minimizes false negatives while improving the accuracy of spam message identification.
# 
# Additionally, the LSTM model's AUC (Area Under the Curve) score is higher, indicating that it performs better overall in differentiating between spam and ham messages. Furthermore, the LSTM model's lower Brier score suggests that its probabilistic predictions are more accurate.
# 
# It's crucial to remember that the Random Forest and SVM models still have their own benefits and function very effectively. For example, they are less likely to mistakenly identify ham transmissions as spam due to their reduced False Positive Rates (FPR) and they are faster to compute than LSTM models.

# # Conclusion

# By evaluating the metrics we can conclude that LSTM outperforms Random Forest and SVM in terms of classification accuracy and other important assessment criteria. However, LSTMs are more complex than Random Forest and SVMs to train. On the other hand, Random Forest and SVM provide consistent performance with a much easier implementation, which makes them the best options when model stability, usability, and quicker training times are more important than optimal performance. 

# # Github Repository Link

# jd849@njit.edu -> [Github Repository Link](https://github.com/jaysheeldodianjit/dodia_jaysheel_finalproject)

# # Screenshots of implementation

# The below image shows a sample of our dataset

# In[120]:


Image("ss/6.png")


# First we load the dataset into a pandas dataframe

# In[121]:


Image("ss/7.png")


# Now we preprocess the dataset columns by pruning redundant columns and renmaing the useful columns to something more comprehnsible

# In[122]:


Image("ss/8.png")


# Now we reduce the size of our dataset to 2,500 Rows by using random sampling. This reduces the amount of data that the models will train on, which in turn reduces the training time.

# In[123]:


Image("ss/9.png")


# Label Encoding is the step where we encode target variables, which are in textual format (red, blue, green...), and convert them into numerical form (0, 1, 2...). In our case we have only 2 categories in the target variable - **Spam** and **Ham**. So there are only 2 numbers after label encoding - **0** and **1**.

# In[124]:


Image("ss/10.png")


# Now we preprocess the dataset for our Machine Learning algorithms
# * We use count vectorizer which converts texts to a matrix of token count.
# * This is done because ML models do not train on text data but rather the numerical format of the text data.

# In[125]:


Image("ss/11.png")


# Now we define functions to calculate the the metrics. The functions defined are:
# 1. **create_cm** - This function creates and returns a confusion matrix of the **tp, tn, fp and fn**.
# 2. **calc_bss** - This function is used to calculate and return the **Brier Skill Score**.
# 3. **calc_all_metrics** - This function is used to calculate and return all the metrics of the dataset. The functions **create_cm** and **calc_bss** are called over here. This function then returns the array of all the metrics that are calculated. All the metrics calculated are:
#    * TP - Number of True Positives
#    * FP - Number of False Positives
#    * TN - Number of True Negatives
#    * FN - Number of False Negatives
#    * TPR - True Positive Rate
#    * FPR - False Positive Rate
#    * TNR - True Negative Rate
#    * FNR - False Negative Rate
#    * Accuracy Score
#    * Precision Score
#    * Recall Score
#    * F1 Score
#    * Brier Score
#    * ROC AUC Score
#    * BSS - Brier Skill Score
# 

# In[126]:


Image("ss/12.png")


# In[127]:


Image("ss/13.png")


# The plotting function is defined to plot the AUC-ROC curve for each fold of the model.

# In[128]:


Image("ss/14.png")


# The dataset is divided into 10 equal parts (called folds), where each fold is used once as a test set while the remaining 9 folds are used as the training set. This is achieved using KFold from sklearn.model_selection.

# In[129]:


Image("ss/15.png")


# The Random Forest Classifier is trained on the dataset and the metrics are calculated for each fold. The metrics are then stored in a pandas dataframe.

# In[130]:


Image("ss/16.png")


# * Once the Random Forest Classifier is trained, we add the name of the model to the metrics to later differentiate between the models when stored in the dataframe.
# * We also call the plotting function to plot the AUC-ROC curve for each fold of the model based on the metrics calculated.

# In[131]:


Image("ss/17.png")


# Now we train the SVM model. SVM is trained on the dataset and the metrics are calculated for each fold. The metrics are then stored in an array which is later used to create a dataframe and tabulate our results.

# In[132]:


Image("ss/18.png")


# Then we follow the same steps for SVM model as we did for Random Forest Classifier. We add the name of the model to the metrics to later differentiate between the models when stored in the dataframe. We also call the plotting function to plot the AUC-ROC curve for each fold of the model based on the metrics calculated.

# In[133]:


Image("ss/19.png")


# <!-- Preprocess for lstm -->
# * Now we preprocess the dataset for LSTM. We break the text into tokens and make them suitable for training an LSTM.
# * We use Tokenizer from `keras.preprocessing.text` to convert the text into tokens.
# * We use pad_sequences from `keras.preprocessing.sequence` to pad the sequences to the equal length.

# In[134]:


Image("ss/20.png")


# Now we definte a function to build the LSTM model. The function is defined as follows:
# * The function takes in the input and we define the input_shape from the input. This is binary classification so the output contains only 1 neuron which will output - 0 or 1.
# * The function then builds the LSTM model using the Sequential API from Keras.
# * The model consists of an Embedding layer, LSTM layer, Dense layer and an output layer.
# * The model is then compiled using the Adam optimizer and binary crossentropy loss function.

# In[135]:


Image("ss/21.png")


# Then we train our LSTM model on the dataset. The model is trained on the dataset and the metrics are calculated for each fold. The metrics are then stored in an array which is later used to create a dataframe and tabulate our results.

# In[136]:


Image("ss/22.png")


# Then we follow the same steps for LSTM model as we did for Random Forest Classifier and SVM. We add the name of the model to the metrics to later differentiate between the models when stored in the dataframe. We also call the plotting function to plot the AUC-ROC curve for each fold of the model based on the metrics calculated.

# In[137]:


Image("ss/23.png")


# Once all the models are trained and the metrics are calculated, we used the stored metrics data into table format using pandas dataframe. The metrics are stored in the dataframe in the following format:
# * Fold Number
# * Model Name
# * TP
# * FP
# * TN
# * FN
# * TPR
# * FPR
# * TNR
# * FNR
# * Accuracy
# * Precision
# * Recall
# * F1 Score
# * Brier Score
# * ROC AUC Score
# * BSS

# In[138]:


Image("ss/24.png")


# We then group the metrics by their ModelName and calculate the mean of all the metrics in the dataframe. This gives us an average performance of the model across all the folds.

# In[139]:


Image("ss/25.png")


# Function print_all_metrics is used to print the mean of all metrics for each model in the dataframe.

# In[140]:


Image("ss/26.png")


# * Now we plot the metrics of Random Forest Classifier by their each folds. The metrics are plotted in the form of a line graph.
# * This is done to visualize the performance of the model across all the folds.
# * This is also done for SVM and LSTM models.

# In[141]:


Image("ss/27.png")


# Finally, we plot a boxplot to compare the performance of all the models, giving us a highs, lows, outliers, etc. This is done to compare the performance of the models and see which model performs the best.

# In[142]:


Image("ss/28.png")


# # Python Script Execution Screenshots

# In[147]:


Image("ss/29.png")


# In[148]:


Image("ss/30.png")


# In[149]:


Image("ss/31.png")


# In[150]:


Image("ss/32.png")


# In[151]:


Image("ss/33.png")


# In[152]:


Image("ss/34.png")


# In[153]:


Image("ss/35.png")


# # Output

# * When executing the code in terminal we see all the metrics that are calculated for each fold when training each model on 10 Fold.
# * We successfully train the model on all the folds and generating graphs for ROC Curve
# * We also print the comparison of mean of metrics for all the models in our terminal. Which shows mean of a metric across all 3 models

# # Other

# The source code (.py file) and data sets (.csv files) will be attached to the zip file.
# 
# Link to GitHub repository
# https://github.com/jaysheeldodianjit/dodia_jaysheel_finalproject
