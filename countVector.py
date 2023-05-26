# https://dacon.io/en/competitions/official/235946/codeshare/6017

import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Get the list of txt files in the directory

path_dir = "D:/OneDrive - Obigo Inc/문서/LDA"
txt_files = os.listdir(path_dir)


# 파일의 전체경로
txt_file_name = 'US10073191B2.txt'
file_fullPath = path_dir + '/' + txt_file_name


# 파일 내용 읽기
with open(file_fullPath, "rt", encoding="UTF-8") as f:
    file_lines = f.readlines()
    txt = " ".join(file_lines)  #파일내용
    count_vect_file = CountVectorizer(stop_words='english', ngram_range=(1,2))
    dict_word_counts = {}
    if file_lines:
        # 문장을 토큰으로 잘라서 단어목록으로 구성
        word_counts = count_vect_file.fit_transform([txt]).toarray()[0] #2차원배열 반환함
        word_names = count_vect_file.get_feature_names_out()

        for i, word in enumerate(word_names):
            dict_word_counts[word] = word_counts[i]

        print(txt_file_name)
        print(dict_word_counts)


