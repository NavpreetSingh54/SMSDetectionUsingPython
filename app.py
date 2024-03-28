import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
import warnings
warnings.filterwarnings("ignore")
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    num = []
    for i in text:
        if i.isalnum():
            num.append(i)
    text = num[:]
    num.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            num.append(i)
    text = num[:]
    num.clear()
    for i in text:
        num.append(ps.stem(i))
    return " ".join(num)
tfid=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))
st.title("Email/SMS Spam Classifier")
input_sns=st.text_area("Enter The Message")
if st.button('Predict'):
  transform_sms = transform_text(input_sns)
  vector_input = tfid.transform([transform_sms])
  result = model.predict(vector_input)[0]
  if result == 1:
    st.header("Spam")
  else:
    st.header("Not Spam")