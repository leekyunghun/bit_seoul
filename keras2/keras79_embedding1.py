from tensorflow.keras.preprocessing.text import Tokenizer

# text = "나는 울트라 맛있는 밥을 먹었다"
# {'나는': 1, '울트라': 2, '맛있는': 3, '밥을': 4, '먹었다': 5}

# text = "나는 울트라 맛있는 밥을 울트라 먹었다"  # 음절: 1글자, 어절: 띄어쓰기 전까지 한묶음, 형태소: 뜻이 있는 가장 작은 단위(ex: 밥, 맛)
# {'울트라': 1, '나는': 2, '맛있는': 3, '밥을': 4, '먹었다': 5}

text = "나는 진짜 맛있는 밥을 진짜 먹었다"
# {'진짜': 1, '나는': 2, '맛있는': 3, '밥을': 4, '먹었다': 5}

token = Tokenizer()
token.fit_on_texts([text])      # 어절로 나누어짐, 많이 나온 어절순으로 index를 줌

print(token.word_index)     

x = token.texts_to_sequences([text])    

print(x)        # [[2, 1, 3, 4, 1, 5]]

from tensorflow.keras.utils import to_categorical
word_size = len(token.word_index)

x = to_categorical(x, num_classes = word_size + 1)  # one-hot-encoding은 0부터 시작하므로 5를 표현하고싶으면 +1을 해줘야함

print(x)