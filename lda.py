#https://dacon.io/en/competitions/official/235946/codeshare/6017

import os
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Get the list of txt files in the directory

path_dir  = "D:/OneDrive - Obigo Inc/문서/LDA"
txt_files = os.listdir(path_dir)
p_summary_section = re.compile('\【[A-Za-z\s]*SUMMARY[A-Za-z\s]*\】')   #[SUMMARY]정규식
p_not_summary_section = re.compile('\【[A-Za-z\s]+\】')              #[xxx]섹션 정규식

# Merge all the txt files into one text
total_lines = []
#문서별 단어List
dict_words = {}
#문서별 SUMMARY
dict_summary = {}
for txt_file_name in txt_files:
    #파일의 전체경로
    file_fullPath = path_dir + '/' + txt_file_name
    
    #파일 내용 읽기
    with open(file_fullPath, "rt", encoding="UTF-8") as f:
        file_lines = f.readlines()
        total_lines += file_lines
        count_vect_file = CountVectorizer()
        if file_lines:
            #문장을 토큰으로 잘라서 단어목록으로 구성
            fect_vect = count_vect_file.fit_transform(file_lines)
            dict_words[txt_file_name] = count_vect_file.get_feature_names_out()
            #SUMMARY 검색
            isSummarySection = False
            summaryText = ""
            for line in file_lines:
                match_summary_section = p_summary_section.match(line)
                match_not_summary_section = p_not_summary_section.match(line)

                if match_summary_section:
                    isSummarySection = True
                elif match_not_summary_section and isSummarySection:
                    break
                elif isSummarySection:
                    # Summary구간인경우
                    summaryText += line

            if summaryText:
                dict_summary[txt_file_name] = summaryText


# for words in dict_words.values():
#     for word in words:
#         print(word)




# Create a TF-IDF vectorizer
count_vect = CountVectorizer(max_df=0.95, max_features=1000, min_df=2, stop_words='english', ngram_range=(1,2))

# Transform the text into a TF-IDF matrix
fect_vect = count_vect.fit_transform(total_lines)

#CountVectorizer객체 내의 전체 word의 명칭을 get_feature_names()을 통해 추출
feature_names = count_vect.get_feature_names_out()
# print(feature_names)


# Create an LDA model
lda = LatentDirichletAllocation(n_components=5)

# Fit the LDA model to the TF-IDF matrix
lda.fit(fect_vect)

def display_topics(model, feature_names, no_top_words):
    for topic_index, topic in enumerate(model.components_):
        print('Topic #', topic_index)

        #components_ array에서 가장 값이 큰 순으로 정렬했을 때, 그 값의 array인덱스를 반환.
        topic_word_indexes = topic.argsort()[::-1]
        topic_word_index=topic_word_indexes[:no_top_words]

        #top_indexes대상인 인덱스별로 feature_names에 해당하는 word feature 추출 후 join concat
        feature_concat = ' '.join([feature_names[i] for i in topic_word_index])
        print(feature_concat)


display_topics(lda, feature_names, 10)


