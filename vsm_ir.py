import sys
import math
import json
import os
from lxml import etree
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import numpy as np


porter_stemmer = PorterStemmer()
TF_IDF = "tf-idf"
COUNT = "count"
TOTAL_FILES = "total_files"
inverted_index = {}
THRESHOLD = 0.08


def convert_lower_case(data):
    return np.char.lower(data)


def remove_stop_words(data):
    stop_words = stopwords.words('english')
    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w
    return new_text


def remove_punctuation(data):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    return data


def remove_apostrophe(data):
    return np.char.replace(data, "'", "")


def stemming(data):
    stemmer = PorterStemmer()

    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + stemmer.stem(w)
    return new_text


# Credit: https://github.com/williamscott701/Information-Retrieval/blob/master/2.%20TF-IDF%20Ranking%20-%20Cosine%20Similarity%2C%20Matching%20Score/TF-IDF.ipynb
def preprocess(data):
    data = remove_punctuation(data)  # remove comma seperately
    data = convert_lower_case(data)
    data = remove_stop_words(data)
    data = stemming(data)
    data = data.strip()
    return data


def parse_file(root):
    for record in root.xpath(".//RECORD"):
        record_num = (record.xpath("./RECORDNUM/text()"))[0].lstrip("0").rstrip()
        text = ""
        if len(record.xpath(".//TITLE/text()")) > 0:
            text += record.xpath(".//TITLE/text()")[0].replace("\n", " ").replace(".", "") + " "
        if len(record.xpath(".//EXTRACT/text()")) > 0:
            text += record.xpath(".//EXTRACT/text()")[0].replace("\n", " ").replace(".", "") + " "
        if len(record.xpath(".//ABSTRACT/text()")) > 0:
            text += record.xpath(".//ABSTRACT/text()")[0].replace("\n", " ").replace(".", "") + " "
        if len(record.xpath(".//TOPIC/text()")) > 0:
            text += record.xpath(".//TOPIC/text()")[0].replace("\n", " ").replace(".", "") + " "
        tokens_after_preprocess = preprocess(text)
        for token in tokens_after_preprocess.split(" "):
            if len(token) == 0:
                continue
            if token in inverted_index:
                if record_num in inverted_index[token]:
                    inverted_index[token][record_num][COUNT] += 1
                else:
                    inverted_index[token][record_num] = {COUNT: 1}
            else:
                inverted_index[token] = {record_num: {COUNT: 1}}


def calc_tf_idf():
    max_freq = {} # denominator in tf formula
    for word in inverted_index:
        for record_num in inverted_index[word]:
            if record_num in max_freq:
                if max_freq[record_num] < inverted_index[word][record_num][COUNT]:
                    max_freq[record_num] = inverted_index[word][record_num][COUNT]
            else:
                max_freq[record_num] = inverted_index[word][record_num][COUNT]

    # calc
    num_of_files = len(max_freq)
    inverted_index[TOTAL_FILES] = num_of_files
    for word in inverted_index:
        if word == TOTAL_FILES:
            continue
        for record_num in inverted_index[word]:
            tf = inverted_index[word][record_num][COUNT] / max_freq[record_num]
            idf = math.log(num_of_files / len(inverted_index[word]), 2)
            inverted_index[word][record_num][TF_IDF] = tf * idf


def build_vocabulary(path):
    for file in os.listdir(path):
        root = etree.parse(path + "\\" + file)
        parse_file(root)
    calc_tf_idf()

    json_file = json.dumps(inverted_index)
    with open("vsm_inverted_index.json", "w") as outfile:
        outfile.write(json_file)
    outfile.close()


def get_weighted_question(question):
    data = preprocess(question)
    question_vec = {}
    max_freq = 0  # denominator in tf formula
    for word in data.split(" "):
        if word == '':
            continue
        if word in question_vec:
            question_vec[word] += 1
        else:
            question_vec[word] = 1
        if max_freq < question_vec[word]:
            max_freq = question_vec[word]

    weighted_question = {}
    for word in question_vec:
        tf = question_vec[word] / max_freq
        if word in inverted_index:
            idf = math.log((inverted_index[TOTAL_FILES] + 1) / len(inverted_index[word]), 2)
        else:
            idf = math.log((inverted_index[TOTAL_FILES] + 1), 2)
        weighted_question[word] = tf * idf
    return weighted_question, data


def get_relevant_docs(question):
    weighted_question, data = get_weighted_question(question)
    data = data.strip()

    # Get all relevant docs by question
    relevant_docs = set()
    for word in data.split(" "):
        if word not in inverted_index:
            continue
        relevant_docs.update(list(inverted_index[word].keys()))

    # calc tf-idf for all relevant docs
    tfidf_docs = {}
    for word in inverted_index:
        if word == TOTAL_FILES:
            continue
        for record_num in inverted_index[word]:
            if record_num in relevant_docs:
                if record_num in tfidf_docs:
                    tfidf_docs[record_num] += inverted_index[word][record_num][TF_IDF] ** 2
                else:
                    tfidf_docs[record_num] = inverted_index[word][record_num][TF_IDF] ** 2

    question_vector_weight = 0
    doc_scores = {}
    for word in weighted_question:
        question_vector_weight += weighted_question[word]**2
        if word in inverted_index:
            for record_num in inverted_index[word]:
                if record_num in doc_scores:
                    doc_scores[record_num] += inverted_index[word][record_num][TF_IDF] * weighted_question[word]
                else:
                    doc_scores[record_num] = inverted_index[word][record_num][TF_IDF] * weighted_question[word]

    # Normalize
    for record_num in doc_scores:
        doc_scores[record_num] = doc_scores[record_num] / math.sqrt(question_vector_weight * tfidf_docs[record_num])

    sorted_docs = [doc.lstrip("0") for doc in sorted(doc_scores, key=lambda k: doc_scores[k], reverse=True) if
                   doc_scores[doc] > THRESHOLD]

    write_file = open('ranked_query_docs.txt', 'w')
    for doc_num in sorted_docs:
        write_file.write(doc_num + '\n')
    write_file.close()
    return sorted_docs


def main():
    args = sys.argv
    if args[1] == 'create_index':
        build_vocabulary(args[2])
    elif args[1] == 'query':
        index_path = args[2]
        question = ' '.join(args[3:])
        with open(index_path) as f:
            data = json.load(f)
        inverted_index.update(data)
        get_relevant_docs(question)


if __name__ == "__main__":
    main()
