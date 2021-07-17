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
from num2words import num2words


porter_stemmer = PorterStemmer()
TF = "tf"
TF_IDF = "tf-idf"
COUNT = "count"
inverted_index = {}


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


def convert_numbers(data):
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        try:
            w = num2words(int(w))
        except:
            a = 0
        new_text = new_text + " " + w
    new_text = np.char.replace(new_text, "-", " ")
    return new_text


def preprocess(data):
    data = convert_lower_case(data)
    data = remove_punctuation(data)  # remove comma seperately
    data = remove_apostrophe(data)
    data = remove_stop_words(data)
    data = convert_numbers(data)
    data = stemming(data)
    data = remove_punctuation(data)
    data = convert_numbers(data)
    data = stemming(data)  # needed again as we need to stem the words
    data = remove_punctuation(data)  # needed again as num2word is giving few hypens and commas fourty-one
    data = remove_stop_words(data)  # needed again as num2word is giving stop words 101 - one hundred and one
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
        for recored_num in inverted_index[word]:
            if recored_num in max_freq:
                if max_freq[recored_num] < inverted_index[word][recored_num][COUNT]:
                    max_freq[recored_num] = inverted_index[word][recored_num][COUNT]
            else:
                max_freq[recored_num] = inverted_index[word][recored_num][COUNT]

    # calc
    num_of_files = len(max_freq)
    for word in inverted_index:
        for recored_num in inverted_index[word]:
            tf = inverted_index[word][recored_num][COUNT] / max_freq[recored_num]
            idf = math.log(num_of_files / len(inverted_index[word]), 2)
            inverted_index[word][recored_num][TF_IDF] = tf * idf


def build_vocabulary(path):
    for file in os.listdir(path):
        root = etree.parse(path + "\\" + file)
        parse_file(root)
    calc_tf_idf()
    print(inverted_index)

    json_file = json.dumps(inverted_index)
    with open("vsm_inverted_index.json", "w") as outfile:
        outfile.write(json_file)
    outfile.close()



def get_relevant_docs(question):



def main():
    args = sys.argv
    if args[1] == 'create_index':
        build_vocabulary(args[2])
    elif args[1] == 'query':
        index_path = args[2]
        question = args[3]
        with open(index_path) as f:
            data = json.load(f)
        inverted_index.update(data)
        get_relevant_docs(question)


if __name__ == "__main__":
    main()