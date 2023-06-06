import streamlit as st
import pandas as pd
import time
import ast
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image

from preprocessing import Preprocessing
preprocessing = Preprocessing()

# Load data
df = pd.read_csv('./dataset/chatgpt_tweets_dataset_preprocessed.csv')
model_performance = pd.read_csv('./dataset/model_performance.csv')
pred_table = pd.read_csv('./dataset/pred_table.csv')

# Load saved model
rf = joblib.load('./saved_model/random_forest.sav')
lgbm = joblib.load('./saved_model/lightgbm.sav')
rf_hyp = joblib.load('./saved_model/random_forest_hyperparameter.sav')
lgbm_hyp = joblib.load('./saved_model/lightgbm_hyperparameter.sav')

# Sidebar
with st.sidebar:
    st.title('Analisis Sentimen Opini Masyarakat terhadap ChatGPT sebagai Aplikasi Natural Language Processing')
    pages = st.selectbox('Pages', ('Overview', 'Visualization', 'Sandbox'))

# Page overview
if pages == 'Overview':
    st.title('Overview')
    st.write('Sampel data')
    st.write(df[['date', 'user_name', 'text', 'sentiment']].sample(10, random_state=8))
    
# Page visualization
if pages == 'Visualization':
    st.title('Visualization')
    
    # Model performance chart
    counts = pred_table.apply(pd.Series.value_counts).reset_index().melt(id_vars='index')
    counts['variable'] = counts['variable'].apply(lambda x: 
                                                  'Y True' if x == 'y_true' else
                                                  'Random Forest' if x == 'random_forest' else
                                                  'LightGBM' if x == 'lightgbm' else
                                                  'Random Forest Hyperparameter' if x == 'random_forest_hyperparameter' else
                                                  'LightGBM Hyperparameter' if x == 'lightgbm_hyperparameter' else x
    )

    fig = px.bar(counts, x='variable', y='value', text=counts['value'], color='index', barmode='group', 
                 labels={'variable': 'Model', 'value': f'Jumlah Prediksi ({counts.value.sum()})', 'index': 'Sentimen'}, 
                 title='Perbandingan Performa Model ketika Memprediksi')

    st.plotly_chart(fig)

    wordcloud = st.selectbox('WordCloud', ('Negatif', 'Netral', 'Positif'))
    if wordcloud == 'Negatif':
        st.image('./reports/negative_wordcloud.png', caption='Negatif')
    elif wordcloud == 'Netral':
        st.image('./reports/neutral_wordcloud.png', caption='Netral')
    elif wordcloud == 'Positif':
        st.image('./reports/positive_wordcloud.png', caption='Positif')
    
# Page sandbox
if pages == 'Sandbox':
    st.title('Sandbox')
    st.write('Ini merupakan sandbox untuk melihat hasil prediksi berdasarkan model yang sudah di training.')
    st.write('Model Performance :')
    st.write(model_performance)
    
    st.header('Predict Test')
    text = st.text_area('Teks untuk analisis sentimen (english)')

    if text:
        text = preprocessing.text_filtering(text)
        token = preprocessing.tokenization(text)
        token = preprocessing.remove_stopwords(token)
        token = preprocessing.stemming(token)
        vector = preprocessing.vector_conversion(token)

        rf_start = time.time()
        rf_pred = preprocessing.label_decoder(rf.predict(vector))
        rf_stop = time.time()

        lgbm_start = time.time()
        lgbm_pred = preprocessing.label_decoder(lgbm.predict(vector))
        lgbm_stop = time.time()

        rf_hyp_start = time.time()
        rf_hyp_pred = preprocessing.label_decoder(rf_hyp.predict(vector))
        rf_hyp_stop = time.time()

        lgbm_hyp_start = time.time()
        lgbm_hyp_pred = preprocessing.label_decoder(lgbm_hyp.predict(vector))
        lgbm_hyp_stop = time.time()

        if rf_pred:
            st.write('Hasil')
            result = pd.DataFrame({
                'model': ['Random Forest', 'LightGBM', 'Random Forest Hyperparameter', 'LightGBM Hyperparameter'],
                'result': [rf_pred, lgbm_pred, rf_hyp_pred, lgbm_hyp_pred],
                'execution_time': [(rf_stop - rf_start), (lgbm_stop - lgbm_start), (rf_hyp_stop - rf_hyp_start), (lgbm_hyp_stop - lgbm_hyp_start)]
            })
            st.write(result)