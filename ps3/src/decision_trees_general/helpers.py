import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return -1 + 1.2*x

def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

if __name__ == '__main__':
    data = [[24, 53, 23, 25, 32, 52, 22, 43, 52, 48], 
         [40, 52, 25, 77, 48, 110, 38, 44, 27, 65], 
        [1, 0, 0, 1, 1, 1, 1, 0, 0, 1]]
    
    plt.figure(figsize=(10, 6))

    plt.scatter([data[0][i]  for i in range(len(data[0])) if data[2][i] == 0], 
                [data[1][i]  for i in range(len(data[0])) if data[2][i] == 0], color='blue', marker='x')
    
    plt.scatter([data[0][i]  for i in range(len(data[0])) if data[2][i] == 1], 
                [data[1][i]  for i in range(len(data[0])) if data[2][i] == 1], color='red', marker='o')
    plt.xlabel('Age')
    plt.ylabel('Salary')

    x = np.linspace(min(data[0]), max(data[0])+1, 100)
    y = f(x)
    plt.plot(x,y, c='black')

    plt.savefig('check.png')

    alpha = -1.2
    beta = 1

    for i in range(len(data[0])):
        print(f'Age: {data[0][i]}, Income: {data[1][i]}, Degree: {data[2][i]}, sign = {sign(alpha*data[0][i]+beta*data[1][i] - 1)}')