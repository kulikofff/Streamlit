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
mname = "facebook/wmt19-en-ru"
tokenizer = FSMTTokenizer.from_pretrained(mname)

def load_model():
    model = FSMTForConditionalGeneration.from_pretrained(mname)
    return model


# Настройка заголовка и текста 
st.title("TRANSLATE DASHBOARD")
st.write("""ПЕРЕВОД C АНГЛИЙСКОГО НА РУССКИЙ С ПОМОЩЬЮ HUGGINGFACE MODEL""")

# Настройка боковой панели
st.sidebar.title("About")
st.sidebar.info(
    """
    This app is Open Source dashboard.
    """
)
st.sidebar.info("Feel free to collaborate and comment on the work. The github link can be found ")

model = load_model()

result = st.text_input('Введите текст на английском:')

if result:
    input = result
    input_ids = tokenizer.encode(input, return_tensors="pt")
    outputs = model.generate(input_ids)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.write('**Результаты перевода:**')
    st.write(decoded)

