from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

model = Sequential()
model.add(Conv2D(10, (2,2), input_shape = (10, 10, 1)))  # 9 x 9 x 10    # convolution 레이어가 많을수록 좋다. 너무 과하지않게
model.add(Conv2D(5, (2,2), padding = 'same'))       # 9 x 9 x 5      # LSTM에서는 LSTM 레이어의 출력shape 차원수가 줄어드는데 
                                                                     # convolution에서는 줄지 않아서 conv2d 레이어를 그냥 붙여도 상관x
model.add(Conv2D(3, (3,3)))       # 7 x 7 x 3
model.add(Conv2D(7, (2,2)))       # 6 x 6 x 7
model.add(MaxPooling2D())         # 3 x 3 x 7
model.add(Flatten())              # Flatten 시켜줌   3 x 3 x7 => (63, )
model.add(Dense(1))

model.summary()

