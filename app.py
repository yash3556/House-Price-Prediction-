from sklearn.ensemble import RandomForestRegressor
import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time
from sklearn.datasets import fetch_california_housing
st.title('🏠House Price prediction using ML')

st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRvQdqIasHkDTf5733FK14z5mPQ18VPhg_R_Q&s')

df = pd.read_csv('house_data.csv')
x = df.iloc[:,:-3]
y = df.iloc[:,-1]

final_x = x
scaler = StandardScaler()
scaled_x = scaler.fit_transform(final_x)

st.sidebar.title('Select House Features: ')
st.sidebar.image('https://cdn.dribbble.com/userupload/20000742/file/original-aaf23458355a156d0cf85b8217a5065a.gif')
all_value = []
for i in final_x:
  min_value = final_x[i].min()
  max_value = final_x[i].max()
  
  result = st.sidebar.slider(f'Select {i} value',min_value,max_value)
  all_value.append(result)

user_x = scaler.transform([all_value])
@st.cache_data
def ml_model(x,y):
  model = RandomForestRegressor()
  model.fit(x,y)
  return model

model = ml_model(scaled_x,y)
house_price = model.predict(user_x)[0]

final_price = round(house_price * 100000,2)
with st.spinner('Predicting House Price'):
  import time
  time.sleep(2)

st.success(f'Estimated House Price is : $ {final_price}')
st.markdown('''**Design and Developed by : Yash Kushwaha**''')









