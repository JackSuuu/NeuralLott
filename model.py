import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt
import random

# lottery dataset
X = np.array([
    [4, 12, 19, 25, 27, 1, 9],
    [19, 20, 26, 29, 30, 3, 4],
    [2, 4, 8, 19, 26, 2, 6],
    [1, 12, 16, 20, 28, 7, 9],
    [7, 9, 11, 31, 32, 4, 8],
    [8, 10, 12, 14, 22, 5, 9],
    [3, 5, 12, 17, 26, 1, 12],
    [1, 2, 8, 18, 27, 4, 8],
    [1, 18, 21, 26, 33, 2, 12],
    [8, 9, 16, 22, 23, 7, 9],
    [13, 18, 20, 26, 28, 4, 8],
    [4, 19, 24, 28, 34, 4, 5],
    [8, 9, 11, 19, 30, 6, 12],
    [9, 12, 15, 30, 34, 5, 6],
    [5, 15, 26, 33, 35, 1, 9],
    [4, 8, 10, 11, 15, 7, 9],
    [23, 26, 27, 29, 32, 5, 8],
    [2, 5, 11, 32, 34, 5, 12],
    [15, 22, 24, 25, 29, 6, 10],
    [3, 10, 11, 20, 22, 7, 11],
    [4, 12, 18, 25, 28, 1, 5],
    [19, 21, 25, 29, 30, 6, 10],
    [2, 5, 9, 20, 26, 1, 12],
    [1, 12, 17, 20, 28, 2, 11],
    [7, 10, 11, 31, 32, 8, 9],
    [8, 10, 11, 14, 22, 5, 8],
    [3, 8, 12, 18, 26, 2, 11],
    [1, 2, 9, 18, 29, 8, 9],
    [1, 18, 21, 26, 30, 5, 8],
    [8, 9, 15, 22, 23, 4, 9],
    [10, 18, 19, 26, 28, 6, 11],
    [5, 19, 24, 28, 34, 5, 6],
    [8, 9, 10, 18, 30, 5, 9],
    [9, 11, 14, 28, 30, 5, 9],
    [5, 14, 23, 33, 34, 1, 5],
    [4, 5, 9, 11, 28, 6, 10],
    [23, 24, 27, 29, 32, 1, 11],
    [10, 18, 19, 26, 28, 2, 11],
    [5, 19, 24, 28, 34, 1, 5],
    [8, 9, 10, 18, 30, 2, 11],
    [9, 11, 14, 28, 30, 8, 9],
    [5, 14, 23, 33, 34, 6, 10],
    [4, 5, 9, 11, 28, 7, 11],
    [5, 19, 24, 28, 34, 1, 5],
    [8, 9, 10, 18, 30, 2, 11],
    [9, 11, 14, 28, 30, 8, 9],
    [5, 14, 23, 33, 34, 6, 10],
    [4, 5, 9, 11, 28, 7, 11],
    [5, 14, 23, 33, 34, 6, 10],
    [5, 14, 23, 33, 34, 6, 10],
])


# Generate Y with 0 or 1 values
Y_true = np.array([[1] for _ in range(22)])
Y_false = np.array([[0] for _ in range(28)])
Y = np.concatenate((Y_true, Y_false), axis=0)


# Plot the data
# Plot X_1 vs X_2
# plt.xlabel('X_1')
# plt.ylabel('X_2')
# plt.title('Data Plot')
# plt.show()

# Model definition
model = tf.keras.Sequential([
    tf.keras.layers.Dense(7, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(2, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output for logistic regression
])

# Compile the model
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001)
)

# Train the model
model.fit(X, Y, epochs=1000, batch_size=1)


# Prediction function
def predict(input_matrix):
    return model.predict(input_matrix)


# Success prob - 中奖倍率
success_prob = 0.7

# Create random number of X and Y array
def predict_once():
    # Generate random lottery number
    ## The first 5 digits
    numbers = list(range(1, 36))
    random.shuffle(numbers)
    numbers = numbers[:5]
    numbers.sort()
    X_Predict_1 = np.array(numbers)

    # The last 2 digits
    numbers = list(range(1, 13))
    random.shuffle(numbers)
    numbers = numbers[:2]
    numbers.sort()
    X_Predict_2 = np.array(numbers)

    # combine the digits
    X_Predict = np.concatenate((X_Predict_1,X_Predict_2), axis=0)
    X_Predict = X_Predict.reshape(1, -1)

    # Make prediction
    prediction = predict(X_Predict)
    if prediction >= success_prob:
        print(f"彩票代码：{X_Predict} 逻辑回归神经网络预测结果 -> 大概率会中")
    else:
        print(f"彩票代码：{X_Predict} 逻辑回归神经网络预测结果 -> 大概率中不了")

    return prediction, X_Predict


prob, number = predict_once()
possible_numbers = []

loop_time = 10000
for i in range(loop_time):
    print(f"中奖概率{prob}")
    if prob >= success_prob:
        possible_numbers.append(number)
    prob, number = predict_once()

print(f"运行模型预测{loop_time}次， 可能中奖序列{possible_numbers}")