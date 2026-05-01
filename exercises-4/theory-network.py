import numpy


def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def forward_propagation(capture, biases, weights):
    input1, input2 = capture
    bias1, bias2, bias3 = biases
    weight1, weight2, weight3, weight4, weight5, weight6 = weights

    weighted_sum1 = (weight1 * input1) + (weight2 * input2) + bias1
    hidden1 = sigmoid(weighted_sum1)

    weighted_sum2 = (weight3 * input1) + (weight4 * input2) + bias2
    hidden2 = sigmoid(weighted_sum2)

    prediction = (weight5 * hidden1) + (weight6 * hidden2) + bias3
    return hidden1, hidden2, prediction


def prediction_offset(target, prediction):
    return 0.5 * (target - prediction) ** 2


def backward_propagation(target, capture, biases, weights, learning_rate):
    input1, input2 = capture
    bias1, bias2, bias3 = biases
    weight1, weight2, weight3, weight4, weight5, weight6 = weights

    hidden1, hidden2, prediction = forward_propagation(capture, biases, weights)

    error_target = prediction - target

    gradient5 = error_target * hidden1
    gradient6 = error_target * hidden2

    error_hidden1 = error_target * weight5 * sigmoid_derivative(hidden1)
    error_hidden2 = error_target * weight6 * sigmoid_derivative(hidden2)

    gradient1 = error_hidden1 * input1
    gradient2 = error_hidden1 * input2
    gradient3 = error_hidden2 * input1
    gradient4 = error_hidden2 * input2

    weight1 -= learning_rate * gradient1
    weight2 -= learning_rate * gradient2
    weight3 -= learning_rate * gradient3
    weight4 -= learning_rate * gradient4
    weight5 -= learning_rate * gradient5
    weight6 -= learning_rate * gradient6

    bias1 -= learning_rate * error_hidden1
    bias2 -= learning_rate * error_hidden2
    bias3 -= learning_rate * error_target

    new_weights = [weight1, weight2, weight3, weight4, weight5, weight6]
    new_biases = [bias1, bias2, bias3]

    return new_weights, new_biases, prediction


def main():
    target = 0.8

    capture = [0.6, 0.1]
    biases = [0.4, -0.2, 0.2]
    weights = [0.2, -0.3, -0.5, 0.1, 0.3, -0.4]

    learning_rate = 0.1

    hidden1, hidden2, prediction = forward_propagation(capture, biases, weights)
    print("Forward output:", prediction)

    new_weights, new_biases, training_prediction = backward_propagation(target, capture, biases, weights, learning_rate)

    print("Output before training:", training_prediction)
    print("Weights after training:", new_weights)

    hidden1, hidden2, new_prediction = forward_propagation(capture, new_biases, new_weights)
    print("Forward output:", new_prediction)


main()
