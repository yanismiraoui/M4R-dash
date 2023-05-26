import base64
import datetime
import io
import pandas as pd
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import os
import re, string
import pickle 
#Essential modules
import pandas as pd
import numpy as np
import time


def load_contents():
    events_file = "./data/events_cleaned.csv"
    stats_file = "./data/model_stats.csv"
    results_data = pd.read_csv(events_file, index_col=0)
    model_stats = pd.read_csv(stats_file, index_col=0)
    return results_data, model_stats



#Convert to lowercase, strip and remove punctuations
def preprocess(text):
    text = text.lower() 
    text = text.strip()  
    text = re.compile('<.*?>').sub('', text) 
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)  
    text = re.sub('\s+', ' ', text)  
    text = re.sub(r'\[[0-9]*\]',' ',text) 
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d',' ',text) 
    text = re.sub(r'\s+',' ',text) 
    return text
