import numpy as np
dataset = np.array(range(1,11))            # (10, )
# # size = 5

print(type(dataset))
def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        aaa.append(subset)

    print(type(aaa))
    return np.array(aaa)

datasets = split_x(dataset, 5)             # (6, 5)
print(datasets)
print(datasets.shape)
print("=====================")

def split_x_2(seq, size):                           
    bbb = []
    for i in range(len(seq) - size + 1):
        for j in range(i, i+1):
            bbb.append(seq[j:j+size, :])
    bbb = np.array(bbb)
    return bbb

result = split_x_2(datasets, 2)            # (5, 2, 5)
print(result)
print(result.shape)

