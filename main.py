import streamlit as st
import pandas as pd
import numpy as np
import pickle
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split

## Tab bar
st.set_page_config(
  page_title = "Heart Disease",
  page_icon = ":heart:"
)

df = pd.read_csv('dataset\dfClean.csv')
X = df.drop("target",axis=1)
y = df['target']

## Preprocessing
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X,y)

scaler = MinMaxScaler()
X_smote_norm = scaler.fit_transform(X_smote)

# Data train dan test (untuk oversampling + normalization )
# X_train_normal, X_test_normal, y_train_normal, y_test_normal = train_test_split(X_smote_norm, y_smote, random_state=42,stratify=y_smote)

# knn_model = KNeighborsClassifier()
# param_grid = {
#     "n_neighbors": range(3,21),
#     "metric":["euclidean","manhattan","chebyshev"],
#     "weights":["uniform","distance"],
#     "algorithm":["auto","ball_tree","kd_three"],
#     "leaf_size":range(10,61),
# }

# knn_model = RandomizedSearchCV(estimator=knn_model,param_distributions=param_grid, n_iter=100, scoring="accuracy", cv=5)

# knn_model.fit(X_train_normal,y_train_normal)

model = pickle.load(open("model/knn_model.pkl","rb"))

y_pred = model.predict(X_smote_norm)
# y_pred_knn = knn_model.predict(X_test_normal)
# accuracy = accuracy_score(y_test_normal,y_pred_knn)
accuracy = accuracy_score(y_smote,y_pred)
accuracy = round((accuracy * 100),2)

## Main Page
st.title(":red[Heart Disease] Classification")
st.write(f"Class: {len(np.unique(y))}")
st.write("Class[0] = :green[Healty]")
st.write("Class[1] = :orange[Level 1] || Class[2] = :orange[Level 2]")
st.write("Class[3] = :red[Level 3] ||  Class[4] = :red[Level 4]")
st.write(f'Accuracy = ',accuracy,'%')

## Tabs
tab1, tab2 = st.tabs(["Single-predict","Multi-predict"])

with tab1 :
  ## User Input Sidebar
  st.sidebar.header("User input")

  age = st.sidebar.number_input(label=":violet[Age]", 
                                min_value=df['age'].min(),
                                max_value=df['age'].max()
                              )
  st.sidebar.write(f":orange[Min]: {df['age'].min()} :red[Max]: {df['age'].max()}")
  st.sidebar.write("")

  sex_sb = st.sidebar.selectbox(label=":violet[Sex]",
                            options=['Male','Female'])
  if sex_sb == "Male":
    sex = 1
  elif sex_sb == "Female":
    sex = 0
  st.sidebar.write("")

  cp_sb = st.sidebar.selectbox(label=":violet[Chest pain type]",
                            options=['Typical angina','Atypical angina','Non-anginal pain','Asymptomatic'])
  if cp_sb == "Typical angina":
    cp = 1
  elif cp_sb == "Atypical angina":
    cp = 2
  elif cp_sb == "Non-anginal pain":
    cp = 3
  elif cp_sb == "Asymptomatic":
    cp = 4

  trestbps = st.sidebar.number_input(label=':violet[Resting blood pressure]',
                                        min_value=df['trestbps'].min(),
                                        max_value=df['trestbps'].max())
  st.sidebar.write(f":orange[Min]: {df['trestbps'].min()} :red[Max]: {df['trestbps'].max()}")
  st.sidebar.write("")

  chol = st.sidebar.number_input(label=':violet[Serum cholestoral]',
                                        min_value=df['chol'].min(),
                                        max_value=df['chol'].max())
  st.sidebar.write(f":orange[Min]: {df['chol'].min()} :red[Max]: {df['chol'].max()}")
  st.sidebar.write("")

  fbs_sb = st.sidebar.selectbox(label=":violet[Fasting blood sugar > 120 mg/dl]",
                            options=['True','False'])
  if fbs_sb == "True":
    fbs = 1
  elif fbs_sb == "False":
    fbs = 0
  st.sidebar.write("")

  restecg_sb = st.sidebar.selectbox(label=":violet[Resting electrocardiographic]",
                            options=['Normal','Having ST-T wave abnormality','Showing probable ventricular hypertrophy'])
  if restecg_sb == "Normal":
    restecg = 0
  elif restecg_sb == "Having ST-T wave abnormality":
    restecg = 1
  elif restecg_sb == "Showing probable ventricular hypertrophy":
    restecg = 2
  st.sidebar.write("")

  thalach = st.sidebar.number_input(label=':violet[Maximum heart rate achieved]',
                                        min_value=df['thalach'].min(),
                                        max_value=df['thalach'].max())
  st.sidebar.write(f":orange[Min]: {df['thalach'].min()} :red[Max]: {df['thalach'].max()}")
  st.sidebar.write("")

  exang_sb = st.sidebar.selectbox(label=":violet[Exercise induced angina]",
                            options=['True','False'])
  if exang_sb == "True":
    exang = 1
  elif exang_sb == "False":
    exang = 0
  st.sidebar.write("")

  oldpeak = st.sidebar.number_input(label=':violet[ST depression induced by exercise relative to rest]',
                                        min_value=df['oldpeak'].min(),
                                        max_value=df['oldpeak'].max())
  st.sidebar.write(f":orange[Min]: {df['oldpeak'].min()} :red[Max]: {df['oldpeak'].max()}")
  st.sidebar.write("")

  ## Data
  data = {
    'Age': age,
    'Sex': sex_sb,
    'CP': cp_sb,
    'Trestbps': f"{trestbps} mm Hg",
    'Chol': f"{chol} mg/dl",
    'FBS > 120 mg/dl?': fbs_sb,
    'Restecg': restecg_sb,
    'Thalach': thalach,
    'Exang': exang_sb,
    'Oldpeak': oldpeak,
  }
  
  tabel_df = pd.DataFrame(data,index=['input'])

  st.header("User Input")
  st.write("")
  st.dataframe(tabel_df.iloc[:,:6])
  st.dataframe(tabel_df.iloc[:,6:])

  predict_btn = st.button("Predict",type="primary")
  

  if predict_btn:
    user_input = [[
          age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak
      ]]
    user_input = np.reshape(user_input,(1,-1))
    user_input = scaler.transform(user_input)
    prediction = model.predict(user_input)[0]
    # prediction = knn_model.predict(user_input)[0]
    
    if prediction == 0:
      result = ":green[Healthy]"
    elif prediction == 1:
      result = ":orange[Level 1]"
    elif prediction == 2:
      result = ":orange[Level 2]"
    elif prediction == 3:
      result = ':red[Level 3]'
    elif prediction == 4:
      result = ':red[Level 4]'
    
    st.write("")
    st.subheader("Prediction : ")
    st.subheader(result)

with tab2 :
  st.header("Predict multiple data")

  sample_csv = df.iloc[:5,:-1].to_csv(index=False).encode('utf-8')

  st.write("")
  st.download_button("Download CSV Example", 
                     data = sample_csv,
                     file_name = 'sample_heart_disease_parameters.csv',
                     mime='text/csv')
  
  st.write("")
  file_uploaded = st.file_uploader("Upload a CSV file",type='csv')
  
  try :
    if file_uploaded :
      uploaded_df = pd.read_csv(file_uploaded)
      prediction_arr = model.predict(uploaded_df)
      # prediction_arr = knn_model.predict(uploaded_df)
      
      result_arr =[]

    for prediction in prediction_arr:
      if prediction == 0:
        result = "Healthy"
      elif prediction == 1:
        result = "Level 1"
      elif prediction == 2:
        result = "Level 2"
      elif prediction == 3:
        result = 'Level 3'
      elif prediction == 4:
        result = 'Level 4'
      result_arr.append(result)
      
    uploaded_result = pd.DataFrame({'Prediction Result': result_arr})
  
    col1, col2 = st.columns([1,2])
    
    with col1:
      st.dataframe(uploaded_result)
    with col2:
      st.dataframe(uploaded_df)
      
  except NameError as ne :
    st.warning(f"Silahkan Masukan file csv")



  
