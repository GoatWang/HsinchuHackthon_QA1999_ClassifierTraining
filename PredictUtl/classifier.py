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
    test_sentences = [
        "要如何申請危機家庭兒童及少年委託安置暨收容費用補助？",
        "發現有迷童、棄嬰或查無身分之兒少，需緊急安置者。",
        "發現有兒童及少年(含發展遲緩或身心障礙者)無父母或父母失聯，需緊急送醫者。",
        "老人健保補助之補助對象、補助額度、申請恢復補助之相關規定為何？",
        "長青學苑報名資格、上課地點及洽詢單位",
        "長者預防走失手鍊及身心障礙者防走失手鍊有何不同？"
    ]
    for sen in test_sentences:
        cat = clf.predict_cat(sen)
        print(sen)
        print(cat)
        print("===================")
