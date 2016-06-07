import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)*np.tanh(x)

def logistic(x):
    return 1/(1+np.exp(-x))

def logistic_derivative(x):
    return logistic(x)*(1-logistic(x))

class NeuralNetwork:
    def __init__(self, layers, activation="tanh"):
        if activation == "logistic":
            self.activation = logistic
            self.activation_derivative = logistic_derivative
        elif activation == "tanh":
            self.activation = tanh
            self.activation_derivative = tanh_derivative
        
        self.weights = []
        
        for i in range(1, len(layers)-1):
            self.weights.append((2*np.random.random((layers[i-1]+1, layers[i]+1))-1)*0.25)
            self.weights.append((2*np.random.random((layers[i]+1, layers[i+1]))-1)*0.25)
        
    def fit (self, X, y, learningrate= 0.2, epochs = 10000):
        X = np.atleast_2d(X)
        temp = np.ones([X.shape[0],X.shape[1]+1])
        temp[:, 0:-1] = X  # adding the bias unit to the input layer
        X = temp
        y=np.array(y)
        
        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]]
            
            for l in range(len(self.weights)):
                a.append(self.activation(np.dot(a[l],self.weights[l])))
            error = y[i]-a[-1]
            deltas = [error * self.activation_derivative(a[-1])]
            
            # start backprobagation
            for l in range(len(a)-2, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[1].T)*self.activation_derivative(a[l]))
            deltas.reverse()
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learningrate * layer.T.dot(delta)
                
    def predict(self, x):
        x = np.array(x)
        temp = np.ones(x.shape[0]+1)
        temp[0:-1] = x 
        a = temp
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a

             
        

            