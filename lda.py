#https://dacon.io/en/competitions/official/235946/codeshare/6017

import os
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Get the list of txt files in the directory

path_dir  = "D:/OneDrive - Obigo Inc/문서/LDA"
txt_files = os.listdir(path_dir )


# Merge all the txt files into one text
lines = []
for txt_file in txt_files:
    with open(path_dir  + '/' + txt_file, "rt", encoding="UTF-8") as f:
        print(path_dir  + '/' + txt_file)
        line = f.readlines()
        lines += line
        count_vect2 = CountVectorizer()
        if line:
            fect_vect = count_vect2.fit_transform(line)
            print(count_vect2.get_feature_names_out())



#
# # Create a TF-IDF vectorizer
# count_vect = CountVectorizer(max_df=0.95, max_features=1000, min_df=2, stop_words='english', ngram_range=(1,2))
#
# # Transform the text into a TF-IDF matrix
# fect_vect = count_vect.fit_transform(lines)
#
# #CountVectorizer객체 내의 전체 word의 명칭을 get_feature_names()을 통해 추출
# feature_names = count_vect.get_feature_names_out()
# print(feature_names)
#
#
# # Create an LDA model
# lda = LatentDirichletAllocation(n_components=5)
#
# # Fit the LDA model to the TF-IDF matrix
# lda.fit(fect_vect)
#
#
#
#
# def display_topics(model, feature_names, no_top_words):
#     for topic_index, topic in enumerate(model.components_):
#         print('Topic #', topic_index)
#
#         #components_ array에서 가장 값이 큰 순으로 정렬했을 때, 그 값의 array인덱스를 반환.
#         topic_word_indexes = topic.argsort()[::-1]
#         topic_word_index=topic_word_indexes[:no_top_words]
#
#         #top_indexes대상인 인덱스별로 feature_names에 해당하는 word feature 추출 후 join concat
#         feature_concat = ' '.join([feature_names[i] for i in topic_word_index])
#         print(feature_concat)
#
#
# display_topics(lda, feature_names, 10)


