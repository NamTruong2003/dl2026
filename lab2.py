def readcsv(f):
    file = open(f,"r")
    lines = file.readlines()[1:]
    x = []
    y = []
    for line in lines:
        xi,yi = line.split(",")
        x.append(int(xi))
        y.append(int(yi))
    return x,y

def L(w0,w1,x,y):
    return (1/2)*pow(w1*x + w0 -y,2)

def J(w0,w1,x,y):
    total = 0
    n = len(x)
    for i in range(n):
        total += L(w0,w1,x[i],y[i])
    return total / n

def dw0(w0,w1,x,y):
    total = 0
    n = len(x)
    for i in range(n):
        total += (w1*x[i] + w0 - y[i])
    return total / n

def dw1(w0,w1,x,y):
    total = 0
    n = len(x)
    for i in range(n):
        total += x[i] * (w1*x[i] + w0 - y[i])
    return total / n

def gradient_descent(w0,w1,x,y,r,t):
    epoch = 0
    w0t = w0
    w1t = w1
    old_loss = float('inf')
    new_loss = J(w0t,w1t,x,y) 
    
    while abs(old_loss - new_loss) > t:
        old_loss = new_loss
        w0t = w0t - r * dw0(w0t,w1t,x,y)
        w1t = w1t - r * dw1(w0t,w1t,x,y)
        new_loss = J(w0t,w1t,x,y)
        epoch += 1
        if epoch % 100 == 0:
            print(f"Epoch {epoch} w0:{w0t:.3f} w1:{w1t:.3f} Loss:{new_loss:.3f}")
    return w0t, w1t

x,y = readcsv("lr.csv")
w0 = 0
w1 = 1
print(f"Initial Loss: {J(w0,w1,x,y)}")
gradient_descent(w0,w1,x,y,0.0001,0.00001)
