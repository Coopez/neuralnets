import numpy as np
import math
import pandas as pd

# Code for assignment 1:
#Rosenblatt Perceptron

# Question - what imports would be dine, and what would not be fine
# aka would numpy be fine here?

def gen_data(P,N):
    # P number of vectors, N dimensionality of vectors
    draw = np.random.normal(size= (P,N))
    data = np.zeros((P,N+1))
    data[:,:-1] = draw
    for vector in range(0,P):
        if np.random.rand() >= 0.5:
            data[vector,-1] = 1   
    return data

def loc_potential(w,e):
    return np.dot(w,e[0:-2]*e[-1])

class Perceptron:
    
    def __init__(self,size,n_data):
        self.P = n_data
        self.N = size
        self.weights = [0 for i in range(0,self.N)] # init weights as 0
        self.train_data = gen_data(self.P,self.N)
    
    def rosenblatt(self,example):
        if loc_potential(self.weights,self.train_data[example]) <= 0:
            self.weights = self.weights + (self.train_data[example][0:-2]*self.train_data[example][-1])/self.N
            return 1
        return 0
    
    def training(self,epochs):
        n = 0
        while  n  < epochs:
            E_sum = 0
            for example in range(0,self.P): # exersice says 1 to P, but since this is index, I would start at 0
                E_sum += self.rosenblatt(example)           
                if E_sum == self.P:
                    return "success"
        return "reached_nmax"

def main():
    #Experiments:
    nD =  50 
    nmax = 100
    N = 20
    alpha = [0.75+0.25*i for i in range(0,10)]
    P = alpha *N
    results = []
    for p in P: # do experiment for each P
        suc_sum = 0
        dt_set = 0
        while dt_set < nD:
            model = Perceptron(p,N)
            if model.training(nmax)  == "success":
               suc_sum +=1
            dt_set += 1
        results.append(suc_sum / nD)



# Testing
#gen_data(2,5)