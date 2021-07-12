import sys
import math
import json
import os
from lxml import etree
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

TF = "tf"
TF_IDF = "tf-idf"
inverted_index = {}


def build_vocabulary(path):
    files_number = 0
    for file in os.listdir(path):
        root = etree.parse(path + "\\" + file)
        files_number += parse_file(root, files_number)
    calc_tf_idf(files_number)
    json_file = json.dumps(inverted_index)
    with open("vsm_inverted_index.json", "w") as outfile:
        outfile.write(json_file)
    outfile.close()


def calc_tf_idf(files_number):
    for word in inverted_index:
        print(word)
        n = len(inverted_index[word][TF])
        idf = math.log(files_number / n, 2)
        for doc in inverted_index[word][TF]:
            inverted_index[word][TF_IDF] = {doc: inverted_index[word][TF][doc] * idf}


def parse_file(root, files_number):
    for record in root.xpath(".//RECORD"):
        files_number += 1
        record_num = (record.xpath("./RECORDNUM/text()"))[0].lstrip("0").rstrip()
        text = ""
        if len(record.xpath(".//TITLE/text()")) > 0:
            text += record.xpath(".//TITLE/text()")[0] + " "
        if len(record.xpath(".//EXTRACT/text()")) > 0:
            text += record.xpath(".//EXTRACT/text()")[0] + " "
        if len(record.xpath(".//ABSTRACT/text()")) > 0:
            text += record.xpath(".//ABSTRACT/text()")[0] + " "

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
                if record_num not in inverted_index[token]:
                    inverted_index[token][record_num] = {TF: tf[token]}
            else:
                inverted_index[token] = {record_num: {TF: tf[token]}}
    return files_number


def main():
    args = sys.argv
    if args[1] == 'create_index':
        build_vocabulary(args[2])
    # elif args[1] == 'query':
    #     ask_question(args[2], 'ontology.nt')


if __name__ == "__main__":
    main()



