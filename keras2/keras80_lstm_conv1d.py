from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

docs = ['너무 재밌어요', '참 최고에요', '참 잘 만든 영화에요',
        '추천하고 싶은 영화입니다', '한 번 더 보고 싶네요', '글쎄요', 
        '별로에요', '생각보다 지루해요', '연기가 어색해요', 
        '재미없어요', '너무 재미없다', '참 재밌네요']

# 1. 긍정 1, 부정 0 
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)

x = token.texts_to_sequences(docs)          
print(x)

from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding = 'pre')   # padding을 어절의 뒤에 하고싶으면 post
print(pad_x)        # (12, 5)
pad_x = pad_x.reshape(pad_x.shape[0], pad_x.shape[1], 1)

word_size = len(token.word_index) + 1
print("전체 토큰 사이즈 : ", word_size) # 25, 전체 단어 종류 갯수

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Flatten, Conv1D

model = Sequential()
# model.add(Embedding(25, 10, input_length = 5))        # Embedding(word_size, 아웃풋 노드의 갯수, input의 한 행의 길이)
# model.add(Embedding(25, 10))                          # input_length를 안적어줘도 잘 돌아감, 단어사전의 갯수를 word_size보다 크게 잡아줘도 진행에 상관x 대신 적으면 에러남

model.add(LSTM(32, input_shape = (5, 1), return_sequences = True))
model.add(Conv1D(32, 3))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))               # 긍정, 부정으로 나누기 때문에 이진분류
# Embedding layer에서 input_length를 지정 안해주면 None으로 출력이되는데 None이 되다보니 정확히 알수없어 Dense층에서 shape를 확인x

model.summary()

model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(pad_x, labels, epochs=50, verbose=1)

accuracy = model.evaluate(pad_x, labels)[1] 
print("accuracy : ", accuracy)

predict = model.predict(pad_x)
print(np.round(predict))