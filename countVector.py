from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()

lines = ["나는 밥을 먹는다", "데이터레이크는 무엇인가", "나는 사과냐?", "단어의 뜻이 무엇이냐"]
fect_vect = count_vect.fit_transform(lines)

feature_names = count_vect.get_feature_names_out()
print(feature_names)