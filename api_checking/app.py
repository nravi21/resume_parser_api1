from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
from werkzeug.datastructures import  FileStorage
import numpy as np
import pandas as pd
import numpy as np
import os
import spacy
import spacy_transformers
import nltk
import re
from flask import Flask, request, jsonify
import sys, fitz
#import ResumeModel.flask.Refunc as fc
from fuzzywuzzy import fuzz
import datetime

# model loading
nlp=nlp1=nlp2=nlp3=nlp4nlp5 = spacy.load('./output/final_trained_ner_model')
# nlp2 = spacy.load("./es_core_news_lg-3.1.0/es_core_news_lg/es_core_news_lg-3.1.0")
# #custom trained model
# nlp1=spacy.load('./output/model-best')
# # # synthetic data model
# nlp3=spacy.load('./output/resume_ner_model')

# nlp4=spacy.load('./output/JD')
# nlp5=spacy.load('./output/final_trained_ner_model_17_10')

import re
import magic
from pdfminer.high_level import extract_text
import docx2txt
import pathlib
import re

#<----extracting name using regex function---->
def extract_text_from_uploaded_file(uploaded_file):
    mime = magic.Magic()
    file_type = mime.from_file(uploaded_file)
    file_extension = pathlib.Path(uploaded_file).suffix

    if (file_extension=='.pdf'):
        return extract_text_from_pdf(uploaded_file)
    elif(file_extension=='.docx'):
        return extract_text_from_docx(uploaded_file)
    else:
        return "unsupported format"

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    return extract_text(pdf_path)

# Function to extract text from a DOCX file using docx2txt
def extract_text_from_docx(docx_file_path):
    try:
        text = docx2txt.process(docx_file_path)
        return text
    except Exception as e:
        return f"Error extracting text from DOCX: {e}"

#fuzzy logic
import nltk
#fuzzzy logic
import spacy
from fuzzywuzzy import fuzz

# Load spaCy's English model
# nlp = spacy.load("en_core_web_sm")

# def fuzzy_preprocessing(etext):
#     test=etext.split('\n')
#     text=[]
#     # Specify the delimiter
#     delimiter = ','
#     # Split and update values in the same list
#     for i in range(len(test)):
#         test[i] = test[i].split(delimiter)
#     for i in range(0,len(test)):
#         for j in range (0,len(test[i])):
#             text.append(test[i][j])
#     while("" in text):
#         text.remove("")
#     while(" " in text):
#         text.remove(" ")
#     # while("\t" in text):
#     #     text.remove(" ")
#     return text

# def fuzzy_match(target, strings, threshold):
#     matches = []

#     for string in strings:
#         similarity_score = fuzz.ratio(target, string)
#         if similarity_score >= threshold:
#             matches.append((string, similarity_score))

#     return matches

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('punkt')
# Removing unwanted characters
def clean_text(text):
    # Remove special characters and symbols except for spaces and @
    cleaned_text = re.sub(r'[^a-zA-Z0-9@\s]', '', text)

    # Remove extra spaces and leading/trailing spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    return cleaned_text

# Stopword Removal
def remove_stopwords(text):
    # Tokenize the text
    words = nltk.word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words]

    # Reconstruct the text without stopwords
    stopremove_text = ' '.join(filtered_words)

    return stopremove_text

# Tokenize Text

def tokenize_text(text):
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    return tokens

#Lemmatization
def lemmatize_tokens(tokenized_text):
    # Initialize the WordNet Lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Lemmatize each token
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokenized_text]

    return lemmatized_tokens


def extract_name_from_resume(text):
    name = None
    test=text.split('\n')
    while("" in test):
        test.remove("")
    while(" " in test):
        test.remove(" ")
    fi = re.sub(' +', ' ',test[0]).lower()
    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    match = re.search(pattern, test[0])
    fi1=bool(match)
    if (fi=="curriculum vitae " or fi=="resume " or fi=="resume" or fi=="curriculum vitae" or fi1==True):
        name=test[1]
    else:
        name=test[0]
    return name
    
#<----end---->
def clean_primary_contact_number(number):
    cleaned_number = re.sub('[^0-9]', '', number)
    # Check the length
    if len(cleaned_number) <= 15:
        return cleaned_number
    else:
        return 'Invalid Number'
#<----extracting contact number using regex---->
def extract_contact_number_from_resume(text):
    contact_number = None

    # Use regex pattern to find a potential contact number
    pattern = r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"

    match = re.search(pattern, text)
    if match:
        contact_number = match.group()

    return contact_number
#<----end---->

#<------extracting email_id using regex ----->
def extract_email_from_resume(text):
    email = None

    # Use regex pattern to find a potential email address
    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    match = re.search(pattern, text)
    if match:
        email = match.group()

    return email
#<----end---->

def standardize_date(date_str):
    date_formats = [
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d-%m-%Y",
        "%Y.%m.%d",
        "%d %B %Y",
        "%d %b %Y"

    ]
    if(date_str is None):
         return None
    else:
        for date_format in date_formats:
            try:
                parsed_date = datetime.datetime.strptime(date_str, date_format)
                return parsed_date.strftime("%Y-%m-%d")
            except ValueError:
                continue

        # If none of the formats match, return None
        return None

#<----extracting Dob using regex---->
def extract_date_of_birth(text):
    dob = None

    # Define patterns that are more likely to represent date of birth
    dob_patterns = [
            r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',  # DD/MM/YYYY or DD-MM-YYYY
            r'\b\d{1,2}[-/][A-Za-z]{3,}[-/]\d{2,4}\b',  # DD/Mon/YYYY or DD/Month/YYYY
            r'\b[A-Za-z]{3,}[-/]\d{1,2}[-/]\d{2,4}\b',  # Mon/DD/YYYY or Month/DD/YYYY
            r'\b\d{1,2}\s[A-Za-z]{3,}\s\d{2,4}\b', # DD Mon YYYY or DD Month YYYY
            r'\b\d{1,2}(st|nd|rd|th)\s[A-Za-z]{3,}\s\d{2,4}\b',  # 1st May 1985 or 23rd May 1985
    ]

    # Convert the text to lowercase for case-insensitive matching
    text = text.lower()

    for pattern in dob_patterns:
        dob_match = re.search(pattern, text)
        if dob_match:
            dob = dob_match.group()
            break

    # Additional keyword-based check
    if dob is None:
        dob_keywords = [
                "date of birth", "dob", "birthdate", "birthday"
        ]
        for keyword in dob_keywords:
            if keyword in text:
                # Find the position of the keyword in the text
                keyword_pos = text.find(keyword)
                # Find the end of the line or sentence after the keyword
                end_pos = text.find('\n', keyword_pos)
                if end_pos == -1:
                    end_pos = text.find('.', keyword_pos)

                if end_pos != -1:
                    # Extract the text between the keyword and the end of the line/sentence
                    dob_text = text[keyword_pos + len(keyword):end_pos].strip()
                    # Attempt to extract DOB from the extracted text using regular expressions
                    for pattern in dob_patterns:
                        dob_match = re.search(pattern, dob_text)
                        if dob_match:
                            dob = dob_match.group()
                            break

                    if dob is not None:
                        break

    return dob
#<----end---->


#<----extracting Address using N.E.R---->
def get_Address(text):
    doc = nlp5(text)
    out = []
    out = [ent.text.replace("\n", " ") for ent in list(doc.ents) if (ent.label_ =='CONTACT ADDRESS')]
    return list(set(out))
#<----end---->

#<----extracting Address using N.E.R---->
def get_country(text):
    doc = nlp5(text)
    out = []
    out = [ent.text.replace("\n", " ") for ent in list(doc.ents) if (ent.label_ =='COUNTRY')]
    return list(set(out))

#<----end---->

#<----extracting State and District using Regex---->

def get_state(text):
    doc = nlp5(text)
    out = []
    out = [ent.text.replace("\n", " ") for ent in list(doc.ents) if (ent.label_ =='STATE')]
    return list(set(out))

def get_district(text):
    doc = nlp5(text)
    out = []
    out = [ent.text.replace("\n", " ") for ent in list(doc.ents) if (ent.label_ =='DISTRICT')]
    return list(set(out))



def get_pincode(text):
    doc = nlp5(text)
    out = []
    out = [ent.text.replace("\n", " ") for ent in list(doc.ents) if (ent.label_ =='ZIPCODE')]
    return list(set(out))

  
#<----end---->

def extract_gender(text):
    gender_terms = ['Male', 'Female', 'Non-Binary', 'Transgender']

    gender_list = [term for term in gender_terms if re.search(r'\b' + term + r'\b', text, re.IGNORECASE)]

    if len(gender_list) == 0:
        return None

    if len(gender_list) == 1:
        return gender_list[0]

    highest_score = 0
    likely_gender = ''
    for gender_candidate in gender_list:
        score = fuzz.token_set_ratio('Gender', gender_candidate)
        if score > highest_score:
            highest_score = score
            likely_gender = gender_candidate

    return likely_gender if highest_score > 50 else None

def extract_marital_status(text):
    marital_status_terms = ['Single', 'Married', 'Divorced', 'Widowed', 'Separated']

    marital_status_list = [term for term in marital_status_terms if re.search(r'\b' + term + r'\b', text, re.IGNORECASE)]

    if len(marital_status_list) == 0:
        return None
    if len(marital_status_list) == 1:
        return marital_status_list[0]
    highest_score = 0
    likely_marital_status = ''
    for marital_status_candidate in marital_status_list:
        score = fuzz.token_set_ratio('Marital Status', marital_status_candidate)
        if score > highest_score:
            highest_score = score
            likely_marital_status = marital_status_candidate

    return likely_marital_status if highest_score > 50 else None

def extract_wedding_date(text):
        # Define a regular expression pattern to capture the wedding date
        pattern = r"(?i)\b(wedding date|date of marriage|married on):\s*([\d/\-]+)\b"

        # Search for the wedding date pattern in the resume text
        match = re.search(pattern, text)

        if match:
            return match.group(2).strip()
        else:
            return None

def extract_current_ctc(text):
        ctc = ""
        ctc_pattern = re.compile(r'\b(?:current\s*ctc|ctc)\s*:\s*([\d,.]+)\b', re.IGNORECASE)
        ctc_match = re.search(ctc_pattern, text)
        if ctc_match:
            ctc = ctc_match.group(1)
        return ctc



#work experience

#<----extracting Company Name using N.E.R---->
def get_company(text):
    doc = nlp5(text)
    out = []
    out = [ent.text.replace("\n", " ") for ent in list(doc.ents) if (ent.label_ =='COMPANY NAME')]
    return list(set(out))

#<----extracting designation using N.E.R---->
def get_designation(text):
    doc = nlp5(text)
    out = []
    out = [ent.text.replace("\n", " ") for ent in list(doc.ents) if (ent.label_ =='DESIGNATION')]
    return list(set(out))
#<----end---->

def get_jt(text):
    doc = nlp5(text)
    out = []
    out = [ent.text.replace("\n", " ") for ent in list(doc.ents) if (ent.label_ =='JOB TITLE')]
    return list(set(out))

def get_estartdate(text):
    doc = nlp5(text)
    out = []
    out = [ent.text.replace("\n", " ") for ent in list(doc.ents) if (ent.label_ =='EMPLOYMENT START DATE')]
    return list(set(out))

def get_eenddate(text):
    doc = nlp5(text)
    out = []
    out = [ent.text.replace("\n", " ") for ent in list(doc.ents) if (ent.label_ =='EMPLOYMENT END DATE')]
    return list(set(out))

def get_achievememnt(text):
    doc = nlp5(text)
    out = []
    out = [ent.text.replace("\n", " ") for ent in list(doc.ents) if (ent.label_ =='ACHIEVEMENTS')]
    return list(set(out))

#Education

#<----extracting education using N.E.R---->


def get_education(text):
                  doc = nlp1(text)
                  education = []

                  education = [ent.text.replace("\n", " ") for ent in list(doc.ents) if (ent.label_=='COLLEGE NAME' or ent.label_=='UNIVERSITY')]
                  return list(set(education))
#<----end---->

#<----extracting degree using N.E.R---->
def get_degree(text):

    doc1 = nlp4(text)
    degree = []
    degree = [ent.text.replace("\n", " ") for ent in list(doc1.ents) if (ent.label_ == 'Degree' or ent.label_=='DEGREE')]
    
    doc2 = nlp5(text)
    out = []
    out = [ent.text.replace("\n", " ") for ent in list(doc2.ents) if (ent.label_ =='DEGREE')]
    return list(set(degree))
#<----end---->
def get_course(text):
    doc = nlp5(text)
    out = []
    out = [ent.text.replace("\n", " ") for ent in list(doc.ents) if (ent.label_ =='SPECIALIZATION')]
    return list(set(out))

def get_institutionname(text):
    doc = nlp5(text)
    out = []
    out = [ent.text.replace("\n", " ") for ent in list(doc.ents) if (ent.label_ =='INSTITUTION NAME')]
    return list(set(out))

def get_passedoutyear(text):
    doc = nlp5(text)
    out = []
    out = [ent.text.replace("\n", " ") for ent in list(doc.ents) if (ent.label_ =='PASSED OUT YEAR')]
    return list(set(out))

def get_percentage(text):
    doc = nlp5(text)
    out = []
    out = [ent.text.replace("\n", " ") for ent in list(doc.ents) if (ent.label_ =='PERCENTAGE ACQUIRED')]
    return list(set(out))



#<----extracting awards using N.E.R---->
def get_awards(text):
                  doc = nlp5(text)
                  awards = []
                  awards = [ent.text.replace("\n", " ") for ent in list(doc.ents) if (ent.label_ == 'ACHIEVEMENTS')]
                  return list(set(awards))
#<----end---->

#<----extracting Skills using N.E.R---->
def get_skills(text):
                  
                  doc = nlp1(text)
                  skills1 = []
                  skills1 = [ent.text.replace("\n", " ") for ent in list(doc.ents) if (ent.label_ == 'SKILLS')]

                  doc = nlp4(text)
                  skills2 = []
                  skills2= [ent.text.replace("\n", " ") for ent in list(doc.ents) if (ent.label_ == 'SKILLS')]
                  
                  skills3=skills1+skills2

                  return list(set(skills3))
#<----end---->

#<----extracting Primary Skills using N.E.R---->
def get_pskills(text):
                                  sents=nlp3(text).to_json()
                                  people=[ee for ee in sents['ents'] if ee['label'] == 'Primary Skills (Technical Skills)']
                                  # print(people)
                                  na=[]
                                  for pps in people:
                                      na.append(text[pps['start']-1:pps['end']])
                                  return na
#<----end---->

#<----extracting Secondary Skills using N.E.R---->
def get_sskills(text):
                                  sents=nlp3(text).to_json()
                                  people=[ee for ee in sents['ents'] if ee['label'] == 'Secondary Skills (Soft Skills)']
                                  # print(people)
                                  na=[]
                                  for pps in people:
                                      na.append(text[pps['start']-1:pps['end']])
                                  return na
#<----end---->

#<----extracting Roles and responsibilities using N.E.R---->
def get_rnr(text):
    sents=nlp3(text).to_json()
    people=[ee for ee in sents['ents'] if ee['label'] == 'Volunteer Roles and Responsibilities']
    # print(people)
    na1=[]
    for pps in people:
        na1.append(text[pps['start']-1:pps['end']])

    responsibilities_match = re.search(r'Responsibilities:\s*(.*?)(\n|$)', text, re.IGNORECASE)

    if responsibilities_match:
        return responsibilities_match.group(1).strip()

    highest_score = 0
    likely_responsibilities = ''
    na2=[]
    for line in text.split('\n'):
        score = fuzz.token_set_ratio('Responsibilities', line)
        if score > highest_score:
            highest_score = score
            likely_responsibilities = line
    if highest_score > 50:
          na2.append(likely_responsibilities)
    else:
          None

    job_description_match = re.search(r'Job Description:\s*(.*?)(\n|$)', text, re.IGNORECASE)

    if job_description_match:
        return job_description_match.group(1).strip()

    highest_score = 0
    likely_description = ''
    for line in text.split('\n'):
        score = fuzz.token_set_ratio('Job Description', line)
        if score > highest_score:
            highest_score = score
            likely_description = line
    na3=[]        
    if highest_score > 50:
            na3.append(likely_responsibilities)
    else:
        None

    na=na1+na2+na3
          
    
    return na


def get_certification_name(text):
    doc = nlp4(text)
    out = []
    out = [ent.text.replace("\n", " ") for ent in list(doc.ents) if (ent.label_ =='CERTIFICATION')]
    return out
def get_issuing_organization(text):
    return None
def get_valid_till(text):
    return None
def get_project_title(text):
    return None
def get_duration(text):
    return None
def get_description(text):
    return None
def get_award_title(text):
    return None
def get_date_received(text):
    return None
def get_description(text):
    return None
def get_publication_title(text):
    return None
def get_co_author(text):
    return None
def get_publication_date(text):
    return None
def get_language_id(text):
    return None
def get_language_id(text):
    return None
def get_organization_name(text):
    return None
def get_retirement_date(text):
    return None
def get_role_and_responsibility(text):
    return None
def get_reference_name(text):
    return None
def get_primary_contact_number(text):
    return None
def get_secondary_contact_number(text):
    return None
def get_email_id(text):
    return None
def get_current_organization(text):
    return None

def make_lists_same_length(lists):
    # Find the maximum length among all lists
    max_length = max(len(lst) for lst in lists)

    # Pad or truncate each list to the maximum length
    result_lists = [lst + [None] * (max_length - len(lst)) for lst in lists]

    return result_lists

def limit_string_length250(input_string, max_length=250):
    if input_string!=None:
        if len(input_string) > max_length:
            return input_string[:max_length]
        return input_string
    else:
        return None

def limit_string_length15(input_string, max_length=15):
    if input_string!=None:
        if len(input_string) > max_length:
            return input_string[:max_length]
        return input_string
    else:
        return None

def limit_string_length100(input_string, max_length=100):
    if input_string!=None:
        if len(input_string) > max_length:
            return input_string[:max_length]
        return input_string
    else:
        return None

def limit_string_length1000(input_string, max_length=1000):
    if input_string!=None:
        if len(input_string) > max_length:
            return input_string[:max_length]
        return input_string
    else:
        return None

def eval_first_name(text):
    if text!=None:
        out=limit_string_length250(text)
    else:
        out=None
    return out
def eval_last_name(text):
    if (text!=""):
        out=limit_string_length250(text)
    else:
        out=None
    return out

def eval_primary_addr_1(text):
    if (text!=""):
        out=limit_string_length250(text)
    else:
        out=None
    return out

def eval_primary_addr_1(text):
    if (text!=""):
        out=limit_string_length250(text)
    else:
        out=None
    return out

def eval_primary_addr_2(text):
    if (text!=""):
        out=limit_string_length250(text)
    else:
        out=None
    return out

def eval_current_ctc(text):
    if (text!=""):
        out=limit_string_length250(text)
    else:
        out=None
    return out
def eval_job_title(text):
    if (text!=""):
        out=limit_string_length100(text)
    else:
        out=None
    return out
def eval_company_name(text):
    if (text!=""):
        out=limit_string_length100(text)
    else:
        out=None
    return out

def eval_designation(text):
    if (text!=""):
        out=limit_string_length250(text)
    else:
        out=None
    return out

def eval_job_description(text):
    if (text!=""):
        out=limit_string_length1000(text)
    else:
        out=None
    return out
def eval_achievements(text):
    if (text!=""):
        out=limit_string_length1000(text)
    else:
        out=None
    return out
def eval_roles_and_responsibilities(text):
    if (text!=""):
        out=limit_string_length250(text)
    else:
        out=None
    return out

def eval_university(text):
    if (text!=""):
        out=limit_string_length100(text)
    else:
        out=None
    return out



def eval_certification_name(text):
    if (text!=""):
        out=limit_string_length250(text)
    else:
        out=None
    return out
def eval_issuing_organization(text):
    if (text!=""):
        out=limit_string_length250(text)
    else:
        out=None
    return out

def eval_project_title(text):
    if (text!=""):
        out=limit_string_length250(text)
    else:
        out=None
    return out
def eval_duration(text):
    if (text!=""):
        out=limit_string_length250(text)
    else:
        out=None
    return out
def eval_description(text):
    if (text!=""):
        out=limit_string_length1000(text)
    else:
        out=None
    return out
def eval_award_title(text):
    if (text!=""):
        out=limit_string_length250(text)
    else:
        out=None
    return out

def eval_description(text):
    if (text!=""):
        out=limit_string_length1000(text)
    else:
        out=None
    return out
def eval_publication_title(text):
    if (text!=""):
        out=limit_string_length250(text)
    else:
        out=None
    return out
def eval_co_author(text):
    if (text!=""):
        out=limit_string_length100(text)
    else:
        out=None
    return out


def eval_organization_name(text):
    if (text!=""):
        out=limit_string_length100(text)
    else:
        out=None
    return out

def eval_role_and_responsibility(text):
    if (text!=""):
        out=limit_string_length250(text)
    else:
        out=None
    return out
def eval_reference_name(text):
    if (text!=""):
        out=limit_string_length100(text)
    else:
        out=None
    return out
def eval_primary_contact_number(text):
    if (text!=""):
        out=limit_string_length15(text)
    else:
        out=None
    return out
def eval_secondary_contact_number(text):
    if (text!=""):
        out=limit_string_length15(text)
    else:
        out=None
    return out
def eval_email_address(text):
    if (text!=""):
        out=limit_string_length100(text)
    else:
        out=None
    return out

def eval_current_organization(text):
    if (text!=""):
        out=limit_string_length250(text)
    else:
        out=None
    return out

def eval_first_name(text):
    if (text!=""):
        out=limit_string_length250(text)
    else:
        out=None
    return out
def eval_last_name(text):
    if (text!=""):
        out=limit_string_length250(text)
    else:
        out=None
    return out

def eval_primary_addr_1(text):
    if (text!=""):
        out=limit_string_length250(text)
    else:
        out=None
    return out
def eval_primary_addr_2(text):
    if (text!=""):
        out=limit_string_length250(text)
    else:
        out=None
    return out

def eval_current_ctc(text):
    if (text!=""):
        out=limit_string_length250(text)
    else:
        out=None
    return out
def eval_job_title(text):
    if (text!=""):
        out=limit_string_length100(text)
    else:
        out=None
    return out
def eval_company_name(text):
    if (text!=""):
        out=limit_string_length100(text)
    else:
        out=None
    return out

def eval_designation(text):
    if (text!=""):
        out=limit_string_length250(text)
    else:
        out=None
    return out

def eval_job_description(text):
    if (text!=""):
        out=limit_string_length1000(text)
    else:
        out=None
    return out
def eval_achievements(text):
    if (text!=""):
        out=limit_string_length1000(text)
    else:
        out=None
    return out
def eval_roles_and_responsibilities(text):
    if (text!=""):
        out=limit_string_length250(text)
    else:
        out=None
    return out

def eval_university(text):
    if (text!=""):
        out=limit_string_length100(text)
    else:
        out=None
    return out



def eval_certification_name(text):
    if (text!=""):
        out=limit_string_length250(text)
    else:
        out=None
    return out
def eval_issuing_organization(text):
    if (text!=""):
        out=limit_string_length250(text)
    else:
        out=None
    return out

def eval_project_title(text):
    if (text!=""):
        out=limit_string_length250(text)
    else:
        out=None
    return out
def eval_duration(text):
    if (text!=""):
        out=limit_string_length250(text)
    else:
        out=None
    return out
def eval_description(text):
    if (text!=""):
        out=limit_string_length1000(text)
    else:
        out=None
    return out
def eval_award_title(text):
    if (text!=""):
        out=limit_string_length250(text)
    else:
        out=None
    return out

def eval_description(text):
    if (text!=""):
        out=limit_string_length1000(text)
    else:
        out=None
    return out
def eval_publication_title(text):
    if (text!=""):
        out=limit_string_length250(text)
    else:
        out=None
    return out
def eval_co_author(text):
    if (text!=""):
        out=limit_string_length100(text)
    else:
        out=None
    return out


def eval_organization_name(text):
    if (text!=""):
        out=limit_string_length100(text)
    else:
        out=None
    return out

def eval_role_and_responsibility(text):
    if (text!=""):
        out=limit_string_length250(text)
    else:
        out=None
    return out
def eval_reference_name(text):
    if (text!=""):
        out=limit_string_length100(text)
    else:
        out=None
    return out
def eval_primary_contact_number(text):
    if (text!=""):
        out=limit_string_length15(text)
    else:
        out=None
    return out
def eval_secondary_contact_number(text):
    if (text!=""):
        out=limit_string_length15(text)
    else:
        out=None
    return out
def eval_email_id(text):
    if (text!=""):
        out=limit_string_length100(text)
    else:
        out=None
    return out
def eval_current_organization(text):
    if (text!=""):
        out=limit_string_length250(text)
    else:
        out=None
    return out

app = Flask(__name__)



@app.route('/')
def out2():
    return render_template("login.html")

@app.route('/login')
def outt():
    return render_template("finalout.html")

@app.route('/check1',methods=['GET', 'POST'])
def check1():
    if request.method == 'POST':
        f = request.files['rfile']
        C_id = int(request.form['Company_Id'])
        # f.save((f.filename))
        f.save("./file/{}".format(f.filename))
        f.save("./cv/resume.pdf")


        extracted_text=extract_text_from_uploaded_file("./file/{}".format(f.filename))
        

        #main script

        if __name__ == '__main__':
            text=text2=extracted_text
            text = " "
            text=' '.join(text.split())
            text=text.lower()
            text=text.replace(',', ' ')
            cleaned_text = clean_text(extracted_text)
            stopremove_text = remove_stopwords(cleaned_text)
            tokenized_text = tokenize_text(stopremove_text)
            lemmatized_tokens = lemmatize_tokens(tokenized_text)
            name=extract_name_from_resume(extracted_text)
            dob1 = extract_date_of_birth(text)
            dob=standardize_date(dob1)
            gender = extract_gender(stopremove_text)
        
            contact_number1 = extract_contact_number_from_resume(extracted_text)
            contact_number=clean_primary_contact_number(contact_number1)
            email1 = extract_email_from_resume(extracted_text)
        
            
            cadd1=get_Address(text2)
            country=get_country(text2)
            state=get_state(text2)
            district=get_district(text2)
            pincode=get_pincode(text2)
            m_stat=extract_marital_status(text)



            
            company1=get_company(text2)
            
            designation1=get_designation(text2)
            job_title1=get_jt(text2)
            esd=get_estartdate(text2)
            eed=get_eenddate(text2)
            achi1=get_achievememnt(text2)
            rnr1=get_rnr(text2)

            education=get_education(text2)
            Degree=get_degree(text2)
            course=get_course(text2)
            iname=get_institutionname(text2)
            pyear=get_passedoutyear(text2)
            percen=get_percentage(text2)


            awards=get_awards(text2)
            skills=get_skills(text2)
            pskills=get_pskills(text2)
            sskills=get_sskills(text2)


            #certification
            f_certification_name1=get_certification_name(extracted_text)
            f_issuing_organization1=get_issuing_organization(extracted_text)
            f_valid_till=get_valid_till(extracted_text)

            #project_details
            f_project_title1=get_project_title(extracted_text)
            f_duration=get_duration(extracted_text)
            f_description1=get_description(extracted_text)

            #Awards
            f_award_title1=get_award_title(extracted_text)
            f_date_received=get_date_received(extracted_text)
            f_description1=get_description(extracted_text)

            #publication
            f_publication_title1=get_publication_title(extracted_text)
            f_co_author1=get_co_author(extracted_text)
            f_publication_date=get_publication_date(extracted_text)

            #Language
            f_language_id=get_language_id(extracted_text)
            f_language_id=get_language_id(extracted_text)

            #Volunteer
            f_organization_name1=get_organization_name(extracted_text)
            f_retirement_date=get_retirement_date(extracted_text)
            f_role_and_responsibility1=get_role_and_responsibility(extracted_text)

            #Reference
            f_reference_name1=get_reference_name(extracted_text)
            f_primary_contact_number1=get_primary_contact_number(extracted_text)
            f_secondary_contact_number1=get_secondary_contact_number(extracted_text)
            f_email_id1=get_email_id(extracted_text)
            f_current_organization1=get_current_organization(extracted_text)

            
           
            email=eval_email_address(email1)
            cadd=eval_primary_addr_1(cadd1)
            job_title=[eval_job_title(number) for number in job_title1]
            company=[eval_company_name(number) for number in company1]
            designation=[eval_designation(number) for number in designation1]
            achi=eval_achievements(achi1)
            rnr=eval_roles_and_responsibilities(rnr1)
            f_certification_name=eval_certification_name(f_certification_name1)
            f_issuing_organization=eval_issuing_organization(f_issuing_organization1)
            f_project_title=eval_project_title(f_project_title1)
            f_description=eval_description(f_description1)
            f_award_title=eval_award_title(f_award_title1)
            f_description=eval_description(f_description1)
            f_publication_title=eval_publication_title(f_publication_title1)
            f_co_author=eval_co_author(f_co_author1)
            f_organization_name=eval_organization_name(f_organization_name1)
            f_role_and_responsibility=eval_role_and_responsibility(f_role_and_responsibility1)
            f_reference_name=eval_reference_name(f_reference_name1)
            f_primary_contact_number=eval_primary_contact_number(f_primary_contact_number1)
            f_secondary_contact_number=eval_secondary_contact_number(f_secondary_contact_number1)
            f_email_id=eval_email_id(f_email_id1)
            f_current_organization=eval_current_organization(f_current_organization1)

            
            

            First_name=""
            Last_name=""
            if name:
                x = name.split()
                if(len(x)==1):
                    First_name=x[0]
                    Last_name=""
                elif(len(x)==2):
                    First_name=x[0]
                    Last_name=x[1]
                elif(len(x)==0):
                    First_name=None
                    Last_name=None

            else:
                First_name=None
                Last_name=None

            if contact_number:
                cn=contact_number
            else:
                cn=None

            if email:
                Email=email
            else:
                Email=None

            if dob:
                Dob=dob
            else:
                Dob=None
            
            if gender:
                Gender=gender
            else:
                Gender=None
            
            if cadd:
                C_Add=cadd
            else:
                C_Add=None
            
            if country:
                Country=country
            else:
                Country=None

            if state:
                State=state
            else:
                State=None

            if district:
                District=district
            else:
                District=None
            
            if pincode:
                Pincode=pincode
            else:
                Pincode=None
            
            if m_stat:
                M_stat=m_stat
            else:
                M_stat=None
            

            if company:
                Company=company
            else:
                Company=None
            
            if designation:
                Designation=designation
            else:
                Designation=None
            
            if job_title:
                Job_title=job_title
            else:
                Job_title=None
            
            if esd:
                Esd=esd
            else:
                Esd=None
            
            if eed:
                Eed=eed
            else:
                Eed=None
            
            if achi:
                Achi=achi
            else:
                Achi=None

            if rnr:
                Rnr=rnr
            else:
                Rnr=None

            if education:
                Edu=education
            else:
                Edu=None
            
            if Degree:
                Deg=Degree
            else:
                Deg=["None"]
            
            # Course=list("Not Fount")
            if course:
                Course=course
            else:
                Course=["None"]
            
     
            if iname:
                Iname=iname
            else:
                Iname=["None"]
            
            
            if pyear:
                Pyear=pyear
            else:
                Pyear=["None"]
           
            if percen:
                Percen=percen
            else:
                Percen=["None"]

            if awards:
                Awards=awards
            else:
                Awards=None
            
            if skills:
                Skills=skills
            else:
                Skills=None
            
            if pskills:
                Pskills=pskills
            else:
                Pskills=None

            if sskills:
                Sskills=sskills
            else:
                Sskills=None

            if f_certification_name:
                F_certification_name=f_certification_name
            else:
                F_certification_name=None
                
            if f_issuing_organization:
                F_issuing_organization=f_issuing_organization
            else:
                F_issuing_organization=None
                
            if f_valid_till:
                F_valid_till=f_valid_till
            else:
                F_valid_till=None
                
            if f_project_title:
                F_project_title=f_project_title
            else:
                F_project_title=None
                
            if f_duration:
                F_duration=f_duration
            else:
                F_duration=None
                
            if f_description:
                F_description=f_description
            else:
                F_description=None
                
            if f_award_title:
                F_award_title=f_award_title
            else:
                F_award_title=None
                
            if f_date_received:
                F_date_received=f_date_received
            else:
                F_date_received=None
                
            if f_description:
                F_description=f_description
            else:
                F_description=None
                
            if f_publication_title:
                F_publication_title=f_publication_title
            else:
                F_publication_title=None
                
            if f_co_author:
                F_co_author=f_co_author
            else:
                F_co_author=None
                
            if f_publication_date:
                F_publication_date=f_publication_date
            else:
                F_publication_date=None
                
            if f_language_id:
                F_language_id=f_language_id
            else:
                F_language_id=None
                
            if f_language_id:
                F_language_id=f_language_id
            else:
                F_language_id=None
                
            if f_organization_name:
                F_organization_name=f_organization_name
            else:
                F_organization_name=None
                
            if f_retirement_date:
                F_retirement_date=f_retirement_date
            else:
                F_retirement_date=None
                
            if f_role_and_responsibility:
                F_role_and_responsibility=f_role_and_responsibility
            else:
                F_role_and_responsibility=None
                
            if f_reference_name:
                F_reference_name=f_reference_name
            else:
                F_reference_name=None
                
            if f_primary_contact_number:
                F_primary_contact_number=f_primary_contact_number
            else:
                F_primary_contact_number=None
                
            if f_secondary_contact_number:
                F_secondary_contact_number=f_secondary_contact_number
            else:
                F_secondary_contact_number=None
                
            if f_email_id:
                F_email_id=f_email_id
            else:
                F_email_id=None
                
            if f_current_organization:
                F_current_organization=f_current_organization
            else:
                F_current_organization=None

            
            lists = [Deg,Course,Iname,Pyear,Percen]
            print(lists)
            result_lists = make_lists_same_length(lists)
            Deg1=result_lists[0]
            Course1=result_lists[1]
            Iname1=result_lists[2]
            Pyear1=result_lists[3]
            Percen1=result_lists[4]
            Edu=[]
            for i in range(0,len(Deg1)):
                Edu.append({"Degree":Deg1[i],"Specialization":Course1[i],"Institution_Name":Iname1[i],"Passedout_Year":Pyear1[i],"Percentage":Percen1[i]})
            
            out={"Company_Id":C_id,"Personal_Information":{"Name_First_Name":First_name,"Name_Last_Name":Last_name,"Contact_number":cn,"Email ":Email,"Date of birth":Dob,"Gender":Gender,"Contact Address":C_Add,"Country":Country,"State":State,"District":district,"Pincode":Pincode,"Marital_status":M_stat},
                "Work_Experience":{"Company":Company,"Designation":Designation,"Job_title":Job_title,"Employement_start_date":Esd,"Employement_end_date":Eed,"Achievements":Achi,"Roles and Responsibilities":Rnr},
                "Education":Edu,
                "Skills":{"skills":Skills,"pskills":Pskills,"sskills":Sskills},
                "Certification_details":{"Certification_Name":F_certification_name,"Issuing Organization":F_issuing_organization,"Valid_till":F_valid_till},
                "Project_Details":{"Project Title":F_project_title,"Duration":F_duration,"Description":F_description},
                "Awards_Details":{"Award Title":F_award_title,"Date Received":F_date_received,"Award Description":F_description},
                "Publication_Details":{"Publications Title":F_publication_title,"Co-Author":F_co_author,"Publication Date":F_publication_date},
                "Language_Details":{"Primary Language":F_language_id,"Secondary Language":F_language_id},
                "Volunteer_Details":{"Organization Name":F_organization_name,"Retirement Date":F_retirement_date,"Roles_and_Responsibilities":F_role_and_responsibility},
                "Reference_Details":{"Reference Name":F_reference_name,"Reference Primary Contact":F_primary_contact_number,"Reference Secondary Contact":F_secondary_contact_number,"Reference Email":F_email_id,"Current Organization":F_current_organization}}


            return jsonify(out)

if __name__ == '__main__':
    app.run(debug=True)
    port = int(os.environ.get('PORT',9000))
    app.run(debug=True, host='0.0.0.0', port=port)



