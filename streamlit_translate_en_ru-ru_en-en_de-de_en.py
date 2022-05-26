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
st.sidebar.info("Feel free to collaborate and comment on the work. The github - https://github.com/kulikofff/Streamlit ")

mname_en_ru = "facebook/wmt19-en-ru"
mname_ru_en = "facebook/wmt19-ru-en"
mname_en_de = "facebook/wmt19-en-de"
mname_de_en = "facebook/wmt19-de-en"
tokenizer_en_ru = FSMTTokenizer.from_pretrained(mname_en_ru)
tokenizer_ru_en = FSMTTokenizer.from_pretrained(mname_ru_en)
tokenizer_en_de = FSMTTokenizer.from_pretrained(mname_en_de)
tokenizer_de_en = FSMTTokenizer.from_pretrained(mname_de_en)


def load_model_en_ru():
    model = FSMTForConditionalGeneration.from_pretrained(mname_en_ru)
    return model

def load_model_ru_en():
    model = FSMTForConditionalGeneration.from_pretrained(mname_ru_en)
    return model

def load_model_en_de():
    model = FSMTForConditionalGeneration.from_pretrained(mname_en_de)
    return model

def load_model_de_en():
    model = FSMTForConditionalGeneration.from_pretrained(mname_de_en)
    return model



# EN -> RU

model_en_ru = load_model_en_ru()

result = st.text_input('Введите текст на английском для перевода на русский:')

if result:
    input = result
    input_ids = tokenizer_en_ru.encode(input, return_tensors="pt")
    outputs = model_en_ru.generate(input_ids)
    decoded = tokenizer_en_ru.decode(outputs[0], skip_special_tokens=True)
    st.write('**Результаты перевода на русский:**')
    st.write(decoded)

# RU -> EN

model_ru_en = load_model_ru_en()

result1 = st.text_input('Enter text in Russian for English translation:')

if result1:
    input = result1
    input_ids = tokenizer_ru_en.encode(input, return_tensors="pt")
    outputs = model_ru_en.generate(input_ids)
    decoded = tokenizer_ru_en.decode(outputs[0], skip_special_tokens=True)
    st.write('**The translate result in English:**')
    st.write(decoded)


# EN -> DE

model_en_de = load_model_en_de()

result2 = st.text_input('Geben Sie den englischen Text ein, um ihn ins Russische zu übersetzen, um ihn ins Deutsche zu übersetzen:')

if result2:
    input = result2
    input_ids = tokenizer_en_de.encode(input, return_tensors="pt")
    outputs = model_en_de.generate(input_ids)
    decoded = tokenizer_en_de.decode(outputs[0], skip_special_tokens=True)
    st.write('**Ergebnisse der deutschen Übersetzung:**')
    st.write(decoded)

# DE -> EN

model_de_en = load_model_de_en()

result3 = st.text_input('Enter text in English for translate in Deutsche:')

if result3:
    input = result3
    input_ids = tokenizer_de_en.encode(input, return_tensors="pt")
    outputs = model_de_en.generate(input_ids)
    decoded = tokenizer_de_en.decode(outputs[0], skip_special_tokens=True)
    st.write('**The translate result in English:**')
    st.write(decoded)