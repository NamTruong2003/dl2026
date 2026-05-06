def y(x):
    return x*x
def y_p(x):
    return 2*x  

def gradient_descent(x,lr):
    time = 0
    while y(x) > 0.01:
        x = x - lr * y_p(x)
        time = time + 1
        print(f"Epoch {time} x:{x:.3f} f(x):{y(x):.3f}")
gradient_descent(10,0.01)
