import jieba
import os
jieba.set_dictionary(os.path.join('models', 'dict.txt.big'))

from pymongo import MongoClient
import pandas as pd
import requests
import xgboost
import xgboost as xgb
import json
import numpy as np


class Classifier():

    def __init__(self):
        with open(os.path.join('models', 'cat_mapping'), 'r' , encoding='utf8') as f:
            self.cat_mapping = json.load(f)
        with open(os.path.join('models', 'vectorterms'), 'r' , encoding='utf8') as f:
            self.vectorterms = json.load(f)

    def getcat_mapping(self):
        return self.cat_mapping

    def getvectorterms(self):
        return self.vectorterms

    def predict_cat(self, test_sentence):
        bst = xgb.Booster({'nthread': 4})  # init model
        bst.load_model(os.path.join('models', '20171125 232430246178.model'))  # load data

        # print(test_sentence)
        words = list(jieba.cut(test_sentence, cut_all=False))
        # print(", ".join(words))
        
        allrelated = []
        for keyword in words:
            url = 'http://140.120.13.244:10000/kem/?keyword='+ keyword +'&lang=cht'
            res = requests.get(url)
            related = [term[0] for term in eval(res.text) if term[1] > 0.65]
            allrelated.extend(related)
        words.extend(allrelated)
        # print(", ".join(words))
            
        self_main_list = [0] * len(self.vectorterms)
        for term in words:
            if term in self.vectorterms:
                idx = self.vectorterms.index(term)
                self_main_list[idx] += 1
            
        vector = self_main_list
        cat_num = bst.predict(xgboost.DMatrix(np.array([vector,])))[0]
        # print(cat_num)

        cat = None
        for key, value in self.cat_mapping.items():
            if str(int(cat_num)) == str(value):
                cat = key
        # print(cat)

        return cat

if __name__ == "__main__":
    clf = Classifier()

    def getall(col_name):
        conn = MongoClient()
        db = conn.QA1999
        collection = db[col_name]
        cursor = collection.find({})
        rows = [row for row in cursor]
        df = pd.DataFrame(rows)
        return df

    df_test= getall('taipei')[:100]
    idxs = np.random.choice(df_test.index, size=5)
    test_sentences = df_test.loc[idxs, 'question']

    for sen in test_sentences:
        cat = clf.predict_cat(sen)
        print(sen)
        print(cat)
        print("======")
