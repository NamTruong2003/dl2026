import matplotlib.pyplot as plt

# Read csv
def readcsv(f):
    x = []
    y = []

    with open(f, "r") as file:
        lines = file.readlines()[1:]

        for line in lines:
            xi, yi = line.strip().split(",")

            x.append(float(xi))
            y.append(float(yi))

    return x, y


# Loss function
def L(w0, w1, x, y):
    return 0.5 * (w1 * x + w0 - y) ** 2


def J(w0, w1, x, y):
    total = 0
    n = len(x)

    for i in range(n):
        total += L(w0, w1, x[i], y[i])

    return total / n


# derivative
def dw0(w0, w1, x, y):
    total = 0
    n = len(x)

    for i in range(n):
        total += (w1 * x[i] + w0 - y[i])

    return total / n


def dw1(w0, w1, x, y):
    total = 0
    n = len(x)

    for i in range(n):
        total += x[i] * (w1 * x[i] + w0 - y[i])

    return total / n


#  GRADIENT DESCENT 
def gradient_descent(w0, w1, x, y, r, t, max_epoch=100000):
    epoch = 0

    old_loss = float("inf")
    new_loss = J(w0, w1, x, y)

    while abs(old_loss - new_loss) > t and epoch < max_epoch:

        old_loss = new_loss

        # Compute gradients FIRST
        d0 = dw0(w0, w1, x, y)
        d1 = dw1(w0, w1, x, y)

        # Then update together
        w0 = w0 - r * d0
        w1 = w1 - r * d1

        new_loss = J(w0, w1, x, y)

        epoch += 1

        if epoch % 100 == 0:
            print(
                f"Epoch {epoch} "
                f"w0:{w0:.4f} "
                f"w1:{w1:.4f} "
                f"Loss:{new_loss:.6f}"
            )

    print("\nTraining Finished")
    print(f"Epochs: {epoch}")
    print(f"Final Loss: {new_loss:.6f}")

    return w0, w1


# main
x, y = readcsv("lr.csv")

w0 = 0
w1 = 1

print(f"Initial Loss: {J(w0, w1, x, y):.6f}")

w0, w1 = gradient_descent(
    w0, w1,
    x, y,
    0.0001,
    0.00001
)

print("\nFinal Parameters")
print("w0 =", w0)
print("w1 =", w1)