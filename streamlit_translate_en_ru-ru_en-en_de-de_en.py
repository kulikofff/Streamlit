import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from urllib.request import urlopen
import json
#pip install MosesTokenizer
#pip install transformers
#pip install sacremoses

import io
from PIL import Image
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions

from transformers import FSMTForConditionalGeneration, FSMTTokenizer

# Настройка заголовка и текста 
st.title("TRANSLATE DASHBOARD")
st.write("""ПЕРЕВОДЫ С ПОМОЩЬЮ HUGGINGFACE MODEL""")

# Настройка боковой панели
st.sidebar.title("About")
st.sidebar.info(
    """
    This app is Open Source dashboard by AVK.
    """
)
st.sidebar.info("Feel free to collaborate and comment on the work. The github link can not be found:) ")

mname_en_ru = "facebook/wmt19-en-ru"
mname_ru_en = "facebook/wmt19-ru-en"
tokenizer_en_ru = FSMTTokenizer.from_pretrained(mname_en_ru)
tokenizer_ru_en = FSMTTokenizer.from_pretrained(mname_ru_en)

def load_model_en_ru():
    model = FSMTForConditionalGeneration.from_pretrained(mname_en_ru)
    return model

def load_model_ru_en():
    model = FSMTForConditionalGeneration.from_pretrained(mname_ru_en)
    return model


model_en_ru = load_model_en_ru()

result = st.text_input('Введите текст на английском:')

if result:
    input = result
    input_ids = tokenizer_en_ru.encode(input, return_tensors="pt")
    outputs = model_en_ru.generate(input_ids)
    decoded = tokenizer_en_ru.decode(outputs[0], skip_special_tokens=True)
    st.write('**Результаты перевода на русский:**')
    st.write(decoded)

model_ru_en = load_model_ru_en()

result1 = st.text_input('Введите текст на русском:')

if result1:
    input = result1
    input_ids = tokenizer_ru_en.encode(input, return_tensors="pt")
    outputs = model_ru_en.generate(input_ids)
    decoded = tokenizer_ru_en.decode(outputs[0], skip_special_tokens=True)
    st.write('**Результаты перевода на английский:**')
    st.write(decoded)