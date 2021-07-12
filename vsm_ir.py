import sys
import json
import os
from lxml import etree
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

FREQ = "Frequency"
TF_IDF = "tf-idf"
inverted_index = {}


def main():
    args = sys.argv
    if args[1] == 'create_index':
        build_vocabulary(args[2])
    elif args[1] == 'query':
        ask_question(args[2], 'ontology.nt')


if __name__ == "__main__":
    main()


def build_vocabulary(path):
    for file in os.listdir(path):
        root = etree.parse(path + "\\" + file)


def parse_file(root):
    for record in root.xpath(".//RECORD"):
        record_num = (record.xpath("./RECORDNUM/text()")).lstrip("0").rstrip()
        text = ""
        text += record.xpath(".//TITLE/text()") + " "
        text += record.xpath(".//EXTRACT/text()") + " "
        text += record.xpath(".//ABSTRACT/text()") + " "

        tokens = [word.lower() for word in text.split(" ") if word != ""]

        tokens_wo_stopwords = set()
        for token in tokens:
            if token not in stopwords.words('english'):
                tokens_wo_stopwords.add(token)

        tf = {}
        count_dict = {}
        for token in tokens_wo_stopwords:
            count_dict[token] = tokens.count(token)
        max_val = max(count_dict.values())

        for token in tokens_wo_stopwords:
            tf[token] = count_dict[token] / max_val

        for token in tokens_wo_stopwords:
            if token in inverted_index:
                if record_num in inverted_index[token]:
                    inverted_index[token][record_num][FREQ] += 1
                else:
                    inverted_index[token][record_num] = {FREQ: 1}
            else:
                inverted_index[token] = {record_num: {FREQ: 1}}







