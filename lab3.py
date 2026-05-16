import math
import matplotlib.pyplot as plt

# read csv
def readcsv(f):
    x1 = []
    x2 = []
    y = []

    with open(f, "r") as file:
        lines = file.readlines()[1:]

        for line in lines:
            x1i, x2i, yi = line.strip().split(",")

            x1.append(float(x1i))
            x2.append(float(x2i))
            y.append(int(yi))

    return x1, x2, y


# - SIGMOID 
def sigmoid(z):

    
    return 1 / (1 + math.exp(-z))
    

#  Loss function 
def L(w0, w1, w2, x1, x2, y):
    y_hat = w1 * x1 + w2 * x2 + w0

    return  - (y * math.log(sigmoid(y_hat)) + (1 - y)*math.log(1- sigmoid(y_hat)))
    

def J(w0, w1, w2, x1, x2, y):
    loss = 0

    for i in range(len(y)):
        loss += L(w0, w1, w2, x1[i], x2[i], y[i])

    return loss / len(y)


# gradient descent
def gradients(w0, w1, w2, x1, x2, y):
    d0 = 0
    d1 = 0
    d2 = 0
    
    n = len(y)

    for i in range(n):
        z = w1 * x1[i] + w2 * x2[i] + w0
        

        

        d0 += 1- y[i] - sigmoid(-z)
        d1 += -y[i]*x1[i] + x1[i]*(1-sigmoid(-z))
        d2 += -y[i]*x2[i] + x2[i]*(1-sigmoid(-z))

    return d0 / n, d1 / n, d2 / n


#  GRADIENT DESCENT 
def gradient_descent(w0, w1, w2, x1, x2, y, r, t, max_epoch=1000000):
    epoch = 0

    new_loss = J(w0, w1, w2, x1, x2, y)

    while new_loss > t and epoch < max_epoch:

        d0, d1, d2 = gradients(w0, w1, w2, x1, x2, y)

        w0 -= r * d0
        w1 -= r * d1
        w2 -= r * d2

        new_loss = J(w0, w1, w2, x1, x2, y)

        epoch += 1

        if epoch % 100 == 0:
            print(
                f"Epoch {epoch} "
                f"w0:{w0:.4f} "
                f"w1:{w1:.4f} "
                f"w2:{w2:.4f} "
                f"Loss:{new_loss:.6f}"
            )

    print("\nTraining Finished")
    print(f"Epochs: {epoch}")
    print(f"Final Loss: {new_loss:.6f}")

    return w0, w1, w2

# Predict
def predict(w0, w1, w2, x1, x2):
    z = w1 * x1 + w2 * x2 + w0
    prob = sigmoid(z)

    if prob >= 0.5:
        return 1
    return 0


# Accuracy
def accuracy(w0, w1, w2, x1, x2, y):
    correct = 0

    for i in range(len(y)):
        pred = predict(w0, w1, w2, x1[i], x2[i])

        if pred == y[i]:
            correct += 1

    return correct / len(y)


# main
w0 = 0
w1 = 0
w2 = 0

r = 0.001
t = 0.2

x1, x2, y = readcsv("loan.csv")

w0, w1, w2 = gradient_descent(
    w0, w1, w2,
    x1, x2, y,
    r, t
)

print("\nFinal Weights")
print("w0 =", w0)
print("w1 =", w1)
print("w2 =", w2)

acc = accuracy(w0, w1, w2, x1, x2, y)

print("\nAccuracy:", round(acc * 100, 2), "%")