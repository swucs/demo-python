#https://dacon.io/en/competitions/official/235946/codeshare/6017

import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Get the list of txt files in the directory

path_dir  = "/Users/hanna/data/lda"
txt_files = os.listdir(path_dir)
p_summary_section = re.compile('\【[A-Za-z\s]*SUMMARY[A-Za-z\s]*\】')   #[SUMMARY]정규식
p_not_summary_section = re.compile('\【[A-Za-z\s]+\】')              #[xxx]섹션 정규식

# Merge all the txt files into one text
total_lines = []
#문서별 단어Dictionay
dict_file_words = {}
#문서별 SUMMARY
dict_file_summary = {}
for txt_file_name in txt_files:
    #파일의 전체경로
    file_fullPath = path_dir + '/' + txt_file_name

    if '.txt' not in txt_file_name:
        continue
    
    #파일 내용 읽기
    with open(file_fullPath, "rt", encoding="UTF-8") as f:
        file_lines = f.readlines()
        total_lines += file_lines
        count_vect_file = CountVectorizer(stop_words='english', ngram_range=(1,2))
        txt = " ".join(file_lines)  # 파일내용TEXT
        if file_lines:
            # 문장을 토큰으로 잘라서 단어목록으로 구성
            word_counts = count_vect_file.fit_transform([txt]).toarray()[0]  # 2차원배열 반환함
            word_names = count_vect_file.get_feature_names_out()
            dict_word_counts = {}   #단어별 갯수 DICT
            for i, word in enumerate(word_names):
                dict_word_counts[word] = word_counts[i]
            dict_file_words[txt_file_name] = dict_word_counts
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
                dict_file_summary[txt_file_name] = summaryText


# for words in dict_words.values():
#     for word in words:
#         print(word)


def do_lda(count_vect, sentence_array, topic_count, keyword_count):

    #단어의 갯수가 저장됨
    word_counts = count_vect.fit_transform(sentence_array)

    #CountVectorizer객체 내의 전체 word의 명칭을 get_feature_names()을 통해 추출
    feature_names = count_vect.get_feature_names_out()

    #Create an LDA model
    lda_model = LatentDirichletAllocation(n_components=topic_count)

    #LDA 수행?
    lda_model.fit(word_counts)

    # 토픽별 단어
    dict_topic_words = {}

    for topic_index, topic in enumerate(lda_model.components_):
        # print('Topic #', topic_index)

        #components_ array에서 가장 값이 큰 순으로 정렬했을 때, 그 값의 array인덱스를 반환.
        topic_word_indexes = topic.argsort()[::-1]
        topic_word_index=topic_word_indexes[:keyword_count]

        #top_indexes대상인 인덱스별로 feature_names에 해당하는 word feature 추출 후 join concat
        feature_concat = ' '.join([feature_names[i] for i in topic_word_index])
        # print(feature_concat)
        # print([feature_names[i] for i in topic_word_index])

        dict_topic_words[topic_index] = [feature_names[i] for i in topic_word_index]


    return dict_topic_words


#LDA수행하기(5개의 토픽, 10개의 키워드)
count_vect = CountVectorizer(max_df=0.95, max_features=1000, min_df=2, stop_words='english', ngram_range=(1, 2)) #단어의 갯수를 세기 위한 Vector???
dict_topic_words = do_lda(count_vect, total_lines, 5, 10)

words_topic0 = dict_topic_words[0]
print("0번 토픽 : " + str(words_topic0))

dict_file_word_count_sum = {}
for filename in dict_file_words:
    word_count_sum = 0
    if filename in dict_file_words:
        dict_word_counts = dict_file_words[filename]

        for word in words_topic0:
            if word in dict_word_counts:
                word_count_sum += dict_word_counts[word]
    # print(filename + " : " + str(word_count_sum))
    dict_file_word_count_sum[filename] = word_count_sum


# word 갯수가 많은 문서대로 역정렬한다.
dict_file_word_count_sum = dict(sorted(dict_file_word_count_sum.items(), key=lambda x: x[1], reverse=True))
print(dict_file_word_count_sum)
print("")

#word갯수가 많은 10개 문서에 대해 Summary의 LDA를 돌린다.
PICK_FILE_COUNT = 10
count_vect = CountVectorizer(stop_words='english') #단어의 갯수를 세기 위한 Vector???
i = 0
for filename in dict_file_word_count_sum.keys():
    i = i + 1
    if i > PICK_FILE_COUNT:
        break

    print(filename)

    #Summary가져오기
    if filename in dict_file_summary:
        summaryText = dict_file_summary[filename]

        #LDA 수행
        dict_topic_words_summary = do_lda(count_vect, [summaryText], 5, 10)
        print(dict_topic_words_summary)
        for key, value in dict_topic_words_summary.items():
            print(str(key) + ":" + str(value))

    else:
        print("Summary 없음")

    print("")
