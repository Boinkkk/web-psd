
import streamlit as st
import pandas as pd

st.header("DATASET CHRONIC KIDNEY DISEASE")

st.markdown('PENJELASAN CHRONIC KIDNEY DISEASE (PENYAKIT GINJAL KRONIS')
st.markdown('Penyakit Ginjal Kronis (PGK) / CHRONIC KIDNEY DISEASE (CKD) ditandai dengan adanya kerusakan ginjal atau laju filtrasi glomerulus (GFR) yang diperkirakan kurang dari 60 mL/menit/1,73 m², yang bertahan selama 3 bulan atau lebih. PGK melibatkan kehilangan fungsi ginjal secara progresif, seringkali menyebabkan kebutuhan akan terapi pengganti ginjal, seperti dialisis atau transplantasi. Klasifikasi KDIGO CKD 2012 mempertimbangkan penyebab yang mendasarinya dan mengkategorikan PGK menjadi 6 tahap perkembangan dan 3 tahap proteinuria berdasarkan laju filtrasi glomerulus dan kadar albuminuria. Meskipun penyebab PGK bervariasi, proses penyakit tertentu menunjukkan pola yang serupa.')

st.markdown('Implikasi PGK sangat luas—muncul dari berbagai proses penyakit dan mempengaruhi kesehatan kardiovaskular, fungsi kognitif, metabolisme tulang, anemia, tekanan darah, dan banyak indikator kesehatan lainnya. Deteksi dini PGK adalah langkah pertama dalam mengobatinya, dan berbagai metode untuk mengukur eGFR telah dijelaskan. Baik faktor risiko yang dapat dimodifikasi maupun yang tidak dapat dimodifikasi mempengaruhi perkembangan PGK. Manajemen PGK melibatkan penyesuaian dosis obat sesuai dengan eGFR pasien, mempersiapkan terapi pengganti ginjal, dan mengatasi penyebab yang dapat diubah untuk memperlambat perkembangan penyakit. Kegiatan ini meninjau etiologi, evaluasi, dan manajemen PGK, dengan menekankan peran penting tim perawatan kesehatan interprofesional dalam memberikan perawatan komprehensif. Pendekatan interprofesional berfokus pada faktor risiko yang dapat dimodifikasi dan tidak dapat dimodifikasi untuk mengelola dan mengurangi perkembangan penyakit.')

st.markdown('# **sumber : [CKD](https://www.ncbi.nlm.nih.gov/books/NBK535404/)**')




df = pd.read_csv('kidney_disease.csv')

#Rename column
col={'age': 'age',
     'bp': 'blood_pressure',
     'sg': 'specific_gravity',
     'al': 'albumin',
     'su': 'sugar',
     'rbc': 'red_blood_cells',
     'pc': 'pus_cell',
     'pcc': 'pus_cell_clumps',
     'ba': 'bacteria',
     'bgr': 'blood_glucose_random',
     'bu': 'blood_urea',
     'sc': 'serum_creatinine',
     'sod': 'sodium',
     'pot': 'potassium',
     'hemo': 'hemoglobin',
     'pcv': 'packed_cell_volume',
     'wc': 'white_blood_cell_count',
     'rc': 'red_blood_cell_count',
     'htn': 'hypertension',
     'dm': 'diabetes_mellitus',
     'cad': 'coronary_artery_disease',
     'appet': 'appetite',
     'pe': 'pedal_edema',
     'ane': 'anemia',
     'classification': 'class'}
df.rename(columns=col, inplace=True)
st.dataframe(df)
df[['packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count']] = df[['packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count']].apply(pd.to_numeric, errors='coerce')


# %% [markdown]
st.markdown('#### DATA SET TERSEBUT MEMPUNYAI 25 FITUR DAN 400 DATA RECORDS')

# %% [markdown]
st.markdown('## DEKSRIPSI SETIAP FITUR')

# %%
# from ucimlrepo import fetch_ucirepo 
  
# # fetch dataset 
# chronic_kidney_disease = fetch_ucirepo(id=336) 
  
# # data (as pandas dataframes) 
# X = chronic_kidney_disease.data.features 
# y = chronic_kidney_disease.data.targets 
  
# # variable information 
# var = chronic_kidney_disease.variables
# st.dataframe(var)

st.markdown('### Analisis Deskriptif')
st.dataframe(df.describe().round(3))


st.markdown('### Missing Value')
st.dataframe(df.isna().sum())


st.markdown('### Total Missing Value')
st.text(df.isna().sum().sum())

numerical_features = df.select_dtypes(include=['int64', 'float64']).copy()
numerical_features[['packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count']] = df[['packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count']].apply(pd.to_numeric, errors='coerce')

numerical_features.aggregate(['mean', 'median', 'std', 'max', 'min', 'sum','var']).round(3)


# %%
categorical_features = df.select_dtypes(include=['object']).copy()
mode_df = categorical_features.mode().iloc[0].to_frame().reset_index()
mode_df.columns = ['feature', 'mode']
mode_df

# %%
from sklearn.preprocessing import LabelEncoder

st.markdown('### Label Encoder')

st.code(''' 
        label_encoders = {}
for column in categorical_features.columns:
    le = LabelEncoder()
    categorical_features[column] = le.fit_transform(categorical_features[column].astype(str))
    label_encoders[column] = le

encoded_df = pd.concat([numerical_features, categorical_features], axis=1)

st.dataframe(encoded_df)
        ''')

label_encoders = {}
for column in categorical_features.columns:
    le = LabelEncoder()
    categorical_features[column] = le.fit_transform(categorical_features[column].astype(str))
    label_encoders[column] = le

encoded_df = pd.concat([numerical_features, categorical_features], axis=1)

st.dataframe(encoded_df)






st.markdown('## ANALISIS PERSEBARAN DATA')

# %% [markdown]
st.markdown('### KESEIMBANGAN DATA ANTARA CKD DAN NOT CKD')

# %%
import matplotlib.pyplot as plt

# Count the occurrences of each class
class_counts = df['class'].value_counts()

# Plot the pie chart
plt.figure(figsize=(8, 6))
plt.text(-1.2, 1, f'CKD: {class_counts["ckd"]}\nNot CKD: {class_counts["notckd"]}', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
class_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
plt.title('Distribution of CKD and Not CKD')
plt.ylabel('')
plt.show()

st.pyplot(plt)

# %% [markdown]
# ## DETEKSI OUTLIER

# %%
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

numerical_features = df.select_dtypes(include=['float64', 'int64']).drop(columns=['id'])

numerical_features.fillna(numerical_features.mean(), inplace=True)

categorical_features = df.select_dtypes(include=['object', 'category']).drop(columns=['class'])
label_encoders = {}
for column in categorical_features.columns:
    le = LabelEncoder()
    categorical_features[column] = le.fit_transform(categorical_features[column].astype(str))
    label_encoders[column] = le
    
combined_features = pd.concat([numerical_features, categorical_features], axis=1)

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.075)
lof_scores = lof.fit_predict(combined_features)
lof_negative_outlier_factor = lof.negative_outlier_factor_

df['lof_score'] = lof_negative_outlier_factor

plt.figure(figsize=(10, 6))
colors = ['blue' if score == 1 else 'red' for score in lof_scores]
plt.scatter(df.index, df['lof_score'], c=colors, alpha=0.6)
plt.axhline(y=-1.9, color='green', linestyle='--', linewidth=2, label='Batasan Outlier')
plt.title('LOF Scores for CKD Dataset')
plt.xlabel('Index')
plt.ylabel('LOF Score')
plt.legend()
plt.grid(True)
plt.show()

st.markdown('# PREPROCESSING')

# %% [markdown]
st.markdown('## IMPUTE MISSING VALUE WITH MEAN')

# %%

st.code(''' 
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

categorical_cols = df.select_dtypes(include=['object', 'category']).columns
df[categorical_cols] = df[categorical_cols].apply(lambda x: x.fillna(x.mode()[0]))
        ''')
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

categorical_cols = df.select_dtypes(include=['object', 'category']).columns
df[categorical_cols] = df[categorical_cols].apply(lambda x: x.fillna(x.mode()[0])) #(apply >= function each column) fill na then fill with mode (each column)

# df.drop(columns=['age_range', 'albumin_category', 'pcv_range', 'lof_score'], inplace=True)

st.dataframe(df)

st.markdown('### Jumlah Missing Value Setelah Di lakukan imputasi')
st.dataframe(df.isna().sum())

# # %% [markdown]
# # ## CLEANING OUTLIER

# # %%
# # df['lof_score'] = lof_scores
# # df = df[lof_scores != -1] -> pick lof_score only -1
# st.dataframe(df)
# df.drop(columns=['lof_score'])

# %% [markdown]
st.markdown('## SCALING USING MINMAX')

# %%

st.code(''' 
        from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

columns_to_scale = df.select_dtypes(include=['float64', 'int64']).columns
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

# %%
numerical_features = df.select_dtypes(include=['float64', 'int64']).drop(columns=['id'])

numerical_features.fillna(numerical_features.mean(), inplace=True)

categorical_features = df.select_dtypes(include=['object', 'category']).drop(columns=['class'])
label_encoders = {}
for column in categorical_features.columns:
    le = LabelEncoder()
    categorical_features[column] = le.fit_transform(categorical_features[column].astype(str))
    label_encoders[column] = le
    
df = pd.concat([numerical_features, categorical_features, df['class']], axis=1)
        ''')
from sklearn.preprocessing import MinMaxScaler

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

columns_to_scale = df.select_dtypes(include=['float64', 'int64']).columns
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

# %%
numerical_features = df.select_dtypes(include=['float64', 'int64']).drop(columns=['id'])

numerical_features.fillna(numerical_features.mean(), inplace=True)

categorical_features = df.select_dtypes(include=['object', 'category']).drop(columns=['class'])
label_encoders = {}
for column in categorical_features.columns:
    le = LabelEncoder()
    categorical_features[column] = le.fit_transform(categorical_features[column].astype(str))
    label_encoders[column] = le
    
df = pd.concat([numerical_features, categorical_features, df['class']], axis=1)


df = df.drop(columns=['lof_score'])
# %%
st.dataframe(df)

# %% [markdown]
st.markdown('# SPLITTING DATASET')

# %%

st.code(''' 
        from sklearn.model_selection import train_test_split

df_clean = df

X = df_clean.drop(columns=['class'])
y = df['class']

y = y.str.strip()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42 )

st.markdown('### Training X ')
st.dataframe( X_train)
st.markdown('### Training Y ')
st.dataframe(y_train.to_frame())
st.markdown('### Testing X ')
st.dataframe(X_test)
st.markdown('### Testing Y ')
st.dataframe(y_test.to_frame())
        ''')
from sklearn.model_selection import train_test_split

df_clean = df

X = df_clean.drop(columns=['class'])
y = df['class']

y = y.str.strip()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42 )

st.markdown('### Training X ')
st.dataframe( X_train)
st.markdown('### Training Y ')
st.dataframe(y_train.to_frame())
st.markdown('### Testing X ')
st.dataframe(X_test)
st.markdown('### Testing Y ')
st.dataframe(y_test.to_frame())

# %% [markdown]
st.markdown('## CHECK IMBALANCE IN DATA TRAINING')

# %%
st.code(''' 
        import matplotlib.pyplot as plt

class_counts_train = y_train.value_counts()

plt.figure(figsize=(8, 6))
plt.text(-1.2, 1, f'CKD: {class_counts_train["ckd"]}\nNot CKD: {class_counts_train["notckd"]}', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
class_counts_train.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
plt.title('Distribution of CKD and Not CKD in Training Data')
plt.ylabel('')
plt.show()
st.pyplot(plt)
        ''')
import matplotlib.pyplot as plt

# Count the occurrences of each class in y_train
class_counts_train = y_train.value_counts()

# Plot the pie chart
plt.figure(figsize=(8, 6))
plt.text(-1.2, 1, f'CKD: {class_counts_train["ckd"]}\nNot CKD: {class_counts_train["notckd"]}', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
class_counts_train.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
plt.title('Distribution of CKD and Not CKD in Training Data')
plt.ylabel('')
plt.show()
st.pyplot(plt)

# %% [markdown]
st.markdown('## MENGATSI IMBALANCING DATA DENGAN MENGGUNAKAN METODE SMOTE')

# %% [markdown]
st.markdown('# SMOTE (Synthetic Minority Oversampling Technique) adalah teknik statistik yang digunakan untuk meningkatkan jumlah kasus dalam kumpulan data yang tidak seimbang. SMOTE merupakan pengembangan dari metode oversampling yang membangkitkan sampel baru dari kelas minoritas.') 
st.markdown('SMOTE bekerja dengan cara:')
st.markdown('Memilih sampel secara acak dari kelas minoritas')
st.markdown('Menemukan K-Nearest Neighbor dari sampel yang dipilih')
st.markdown('Menghubungkan sampel yang dipilih ke masing-masing tetangganya menggunakan garis lurus')
st.markdown('SMOTE menghasilkan instans baru dari kasus minoritas yang ada, bukan hanya salinan dari mereka. Pendekatan ini meningkatkan fitur yang tersedia untuk setiap kelas dan membuat sampel lebih umum')

# %%
from imblearn.over_sampling import SMOTE

X_smote, y_smote = SMOTE().fit_resample(X_train, y_train)

class_counts_train = y_smote.value_counts()

# Plot the pie chart
plt.figure(figsize=(8, 6))
plt.text(-1.2, 1, f'CKD: {class_counts_train["ckd"]}\nNot CKD: {class_counts_train["notckd"]}', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
class_counts_train.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
plt.title('Distribution of CKD and Not CKD in Training Data')
plt.ylabel('')
plt.show()

st.markdown('### Persebaran class dataset setelah Proses SMOTE')
st.pyplot(plt)





# %% [markdown]
st.markdown('## MODELLING DATA MENGGUNAKAN NAIVE BAYES')

st.code(''' 
        from sklearn.naive_bayes import GaussianNB
gaussian = GaussianNB()
gaussian.fit(X_smote, y_smote)

predict_gaussian = gaussian.predict(X_test)

result_gaussian = pd.concat([pd.DataFrame(X_test).reset_index(), pd.DataFrame(predict_gaussian)], axis=1)
        
        ''')
# %%
from sklearn.naive_bayes import GaussianNB
gaussian = GaussianNB()
gaussian.fit(X_smote, y_smote)

predict_gaussian = gaussian.predict(X_test)

result_gaussian = pd.concat([pd.DataFrame(X_test).reset_index(), pd.DataFrame(predict_gaussian)], axis=1)

st.markdown('### Hasil Prediksi Gaussian')
st.dataframe(result_gaussian)

# %% [markdown]
st.markdown('# EVALUASI')

# %%

st.code (''' 
         from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, classification_report

precisionScore = precision_score(predict_gaussian, y_test, average='micro')
recallScore = recall_score(predict_gaussian, y_test, average='micro')
f1Score = f1_score(predict_gaussian, y_test, average='micro')
accuracyScore = accuracy_score(predict_gaussian, y_test)

sum_accuracy = pd.DataFrame.from_dict({
    'Precission Score' : [precisionScore],
    'Recall Score' : [recallScore],
    'F1 Score' : [f1Score],
    'Accuracy Score' : [accuracyScore], 
})

print(classification_report(predict_gaussian, y_test))
         ''')

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, classification_report

precisionScore = precision_score(predict_gaussian, y_test, average='micro')
recallScore = recall_score(predict_gaussian, y_test, average='micro')
f1Score = f1_score(predict_gaussian, y_test, average='micro')
accuracyScore = accuracy_score(predict_gaussian, y_test)

sum_accuracy = pd.DataFrame.from_dict({
    'Precission Score' : [precisionScore],
    'Recall Score' : [recallScore],
    'F1 Score' : [f1Score],
    'Accuracy Score' : [accuracyScore], 
})

print(classification_report(predict_gaussian, y_test))

st.markdown('### Hasil Evaluasi')
st.dataframe(sum_accuracy)




# %%
import seaborn as sns
confussionMatrix = confusion_matrix(y_test, predict_gaussian, labels=['ckd', 'notckd'])

plt.subplots(figsize=(10, 8))
sns.heatmap(confussionMatrix, annot=True, fmt='d', cmap='Blues', xticklabels=['CKD', 'Not CKD'], yticklabels=['CKD', 'Not CKD'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

st.markdown('### Confusion Matrix')
st.pyplot(plt)

# %% [markdown]
st.markdown('## MEMBANDINGKAN NAIVE BAYES DENGAN ALGORITMA LAIN')

# %%
st.code(''' 
        from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


accuracys = []
method_list = [KNeighborsClassifier(), DecisionTreeClassifier()]

def classify(method):
    clf = method
    clf.fit(X_smote, y_smote)
    result = clf.predict(X_test)
    acc = accuracy_score(result, y_test)
    print(str(method), acc)

for method in method_list:
    classify(method)
        ''')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


accuracys = []
method_list = [KNeighborsClassifier(), DecisionTreeClassifier()]

st.markdown('### Hasil Metode Lain')

def classify(method):
    clf = method
    clf.fit(X_smote, y_smote)
    result = clf.predict(X_test)
    acc = accuracy_score(result, y_test)
    print(str(method), acc)
    st.markdown(f'{str(method)} : {acc}')

for method in method_list:
    classify(method)
    
    
    
    
    

# %% [markdown]
st.markdown('## CLASIFIKASI MENGGUNAKAN NEURAL) NETWORK')

st.code(''' 
        torch.manual_seed = 30
model = Model()


nn_df = df


y_smote = y_smote.replace('ckd', 0)
y_smote = y_smote.replace('notckd', 1)

y_test = y_test.replace('ckd', 0)
y_test = y_test.replace('notckd', 1)


import numpy as np

X = X_smote.values
y = y_smote.values

X_test = X_test.values
y_test = y_test.values

# To float tensor (X)

X_train = torch.FloatTensor(X)
X_test = torch.FloatTensor(X_test)

# To float tensor (Y)

y_train = torch.LongTensor(y)
y_test = torch.LongTensor(y_test)

st.dataframe(pd.DataFrame(X_train))


#Set measure error of the model, how far prediction of data using Adam Optimezer

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# Train model
# Epochs -> how many turn data to fit to the model

epochs =  100
losses = []

for i in range(epochs):
    y_pred = model.forward(X_train)
    loss = criterion(y_pred, y_train)

    losses.append(loss.detach().numpy())
    
    if i % 10 == 0:
        print(f'Epoch : {i} and loss: {loss}')
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
plt.plot(range(epochs), losses)
plt.ylabel('Loss/Error')
plt.xlabel('Epoch')
plt.show()

# %%
with torch.no_grad():
    y_eval = model.forward(X_test)
    loss = criterion(y_eval, y_test)

# %%
loss

# %%
correct = 0
with torch.no_grad():
    for i, data in enumerate(X_test):
        y_val = model.forward(data)
        
        if y_test[i] == 0:
            x = 'ckd'
        else:
            x = 'notckd'
        
        print(f'{i + 1}) {str(y_val)} \t {x} \t\t {y_val.argmax().item()}')
        
        if y_val.argmax().item() == y_test[i]:
            correct += 1
    print(f'Got Correct {correct}')
        ''')
# %%
import torch
import torch.nn as nn
import torch.nn.functional as F


# %%
# Input Layer = 25 Feature
# 3 HIDDEN LAYER 
# Output layer = 2 Class

class Model(nn.Module):
    def __init__(self, in_features = 24, h1 = 8, h2 = 9, h3 = 10, out_features = 2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2,h3)
        self.out = nn.Linear(h3, out_features)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)
        
        return x
    

# %%
## RANDOMZIER

torch.manual_seed = 30
model = Model()


nn_df = df


y_smote = y_smote.replace('ckd', 0)
y_smote = y_smote.replace('notckd', 1)

y_test = y_test.replace('ckd', 0)
y_test = y_test.replace('notckd', 1)


import numpy as np

X = X_smote.values
y = y_smote.values

X_test = X_test.values
y_test = y_test.values

# To float tensor (X)

X_train = torch.FloatTensor(X)
X_test = torch.FloatTensor(X_test)

# To float tensor (Y)

y_train = torch.LongTensor(y)
y_test = torch.LongTensor(y_test)


#Set measure error of the model, how far prediction of data using Adam Optimezer

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# Train model
# Epochs -> how many turn data to fit to the model

epochs =  100
losses = []

for i in range(epochs):
    y_pred = model.forward(X_train)
    loss = criterion(y_pred, y_train)

    losses.append(loss.detach().numpy())
    
    if i % 10 == 0:
        print(f'Epoch : {i} and loss: {loss}')
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
plt.plot(range(epochs), losses)
plt.ylabel('Loss/Error')
plt.xlabel('Epoch')
plt.show()


# %%
with torch.no_grad():
    y_eval = model.forward(X_test)
    loss = criterion(y_eval, y_test)

# %%
# loss

# %%
correct = 0
with torch.no_grad():
    for i, data in enumerate(X_test):
        y_val = model.forward(data)
        
        if y_test[i] == 0:
            x = 'ckd'
        else:
            x = 'notckd'
        
        print(f'{i + 1}) {str(y_val)} \t {x} \t\t {y_val.argmax().item()}')
        
        st.text(f'{i + 1}) {str(y_val)} \t {x} \t\t {y_val.argmax().item()}')
        
        if y_val.argmax().item() == y_test[i]:
            correct += 1
    print(f'Got Correct {correct}')
    st.text(f'Hasil Prediksi Neural Network : {correct}')



    
    