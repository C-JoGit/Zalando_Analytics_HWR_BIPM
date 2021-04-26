'''
Instructions/Notes

Various functions and libraries for the project are imported or defined in this file, to avoid a lot of information on the project file.

Use conda install (instead of pip install) on your command/terminal prompt to install packages such as : conda install gensim, conda install spacy, conda install python-Levenshtein, conda install etc...

For older python version, import en_core_web_sm works, for newer version - import spacy instead also install this on terminal/command prompt 'python -m spacy download en_core_web_sm' to enable this code


'''

import pandas as pd
import re
from gensim.parsing.preprocessing import STOPWORDS, strip_tags, strip_numeric, strip_punctuation, strip_multiple_whitespaces, remove_stopwords, strip_short, stem_text
import pickle
import spacy # import en_core_web_sm 
import nltk
nltk.download('stopwords')
import os

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import Binarizer
from gensim.corpora import Dictionary
from gensim.models import TfidfModel