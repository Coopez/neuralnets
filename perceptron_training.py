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
    return np.dot(w,e*w[-1])

class Perceptron:
    def __init__(self,size,n_data):
        self.P = n_data
        self.N = size
        self.weights = [0 for i in range(0,self.N)] # init weights as 0
        self.train_data = gen_data(self.P,self.N)
    def training(self,epochs):
        for n in range(0,epochs):
            for example in range(0,self.P): # exersice says 1 to P, but since this is index, I would start at 0
                pass
                # Here update perceptron function
    def rosenblatt(self,example):
        if loc_potential(self.weights[example],self.train_data[example]) <= 0:
            pass
        
        return self.weights







# Testing
#gen_data(2,5)