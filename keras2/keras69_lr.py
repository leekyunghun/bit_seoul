weight = 0.5
input = 0.5
goal_prediction = 0.8
lr = 0.01 # 0.001 # 0.1 / 1 / 0.0001 / 10

for iteration in range(1000):
    prediction = input * weight                            # 목표치에 향해 가는 예측값
    error = (prediction - goal_prediction) ** 2            # 예측값과 목표값과의 오차
    print(iteration)
    print("Error : " + str(error) + "\tPrediction : " + str(prediction))

    up_prediction = input * (weight + lr)                  # 예측값 업데이트 
    up_error = (goal_prediction - up_prediction) ** 2      # 업데이트 한 예측값과 목표값의 오차
    print("up_prediction : ", up_prediction)
    print("up_error : ", up_error, "\n")

    down_prediction = input * (weight - lr)                # 예측값 업데이트 
    down_error = (goal_prediction - down_prediction) ** 2  # 업데이트 한 예측값과 목표값의 오차
    print("down_prediction : ", down_prediction)
    print("down_error : ", down_error, "\n")

    if(down_error < up_error):
        weight = weight - lr
        print("down_error < up_error")
        break                                              # 가중치 업데이트 하면서 목표값을 넘어섰을때 정지

    if(down_error > up_error):              
        weight = weight + lr
        print("down_error > up_error\n")

    print("weight : ", weight, "\n")                       # 가중치의 값 확인