import numpy as np

# Code for assignment 1:
# Rosenblatt Perceptron
# Authors: Sebastian Prehn & Niklas Erdmann, 11.2020

def gen_data(P,N):
    # P number of vectors, N dimensionality of vectors
    # returns a matrix of data vectors with the last column being the labels 1,-1
    draw = np.random.normal(size= (P,N))
    data = np.ones((P,N+1))
    data[:,:-1] = draw
    for vector in range(0,P):
        if np.random.rand() >= 0.5:
            data[vector,-1] = -1   
    return data

def loc_potential(w,e):
    return np.dot(w,[number*e[-1] for number in  e[0:-1]])

class Perceptron:
    def __init__(self,size,n_data):
        self.P = n_data
        self.N = size
        self.weights = np.zeros(self.N) # init weights as 0s
        self.train_data = gen_data(int(self.P),int(self.N))
    
    def rosenblatt(self,example):
        if loc_potential(self.weights,self.train_data[example,:]) <= 0:
            self.weights = self.weights + (self.train_data[example,0:-1]*self.train_data[example,-1])/self.N
            return 0
        return 1
    
    def training(self,epochs):
        n = 1
        while  n  <= epochs:
            E_sum = 1
            for example in range(0,self.P): # exercise says 1 to P, but since this is an index, we start at 0 and go up to, but not including P 
                E_sum += self.rosenblatt(example)       
                if E_sum == self.P:
                    return "success",n-1
            n+=1
        return "reached_nmax",n-1

def main():

    #######################
    ## Experiment Settings:
    nD =  50  # Number of Experiment runs
    nmax = 100 # Maximal Epochs per Experiment
    N = 20 # size of each Sample instance
    print_avgEpochs = True # if True will print out mean epochs to reach results for each P.
    #######################

    alpha = [0.75+0.25*i for i in range(0,10)]
    P = [int(a*N) for a in alpha]
    results = []
    epochs = []
    for p in P: # do experiment for each item in P
        suc_sum = 0
        dt_set = 0
        e = 0
        while dt_set < nD: 
            model = Perceptron(N,p)
            train, epoch = model.training(nmax)
            e += epoch 
            if train  == "success":   
                suc_sum +=1
            dt_set += 1
        results.append(suc_sum / nD)
        epochs.append(e/nD)
    print("Results: " + str(results))
    if print_avgEpochs:
        print("Average number of Epochs for each P: "+ str(epochs))

main()