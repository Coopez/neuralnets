# Code for assignment 1:
# Rosenblatt Perceptron
# Authors: Sebastian Prehn & Niklas Erdmann, 11.2020
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as col
import sys
import ast

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

def plot_results(results,alpha,P,N):
    N_Group = []
    for n in N:
        temp = [n for i in range(0,10)]
        N_Group += temp
    N_Alpha = []
    for i in range(0,len(N)):
        N_Alpha += alpha
    data_unified = {'Probability of lin. Seperability':results,'P/N':N_Alpha, 'N':N_Group}
    data = pd.DataFrame(data_unified)
    sns.set(font_scale=1.3)
    sns.set_style("darkgrid") # ticks or whitegrid darkgrid
    # palette options: flare mako_r Set2 Set1 Set3
    sns.lineplot(data=data, x='P/N', y='Probability of lin. Seperability', hue = 'N',style='N',palette="Set1",hue_norm=col.LogNorm())
    plt.show()

def run_experiment(print_avgEpochs,*parameters):
    if not parameters:
        nD =  50  # Number of Experiment runs
        nmax = 100 # Maximal Epochs per Experiment
        N = [10,20,100] # size of each Sample instance
    else: 
        parameter = parameters[0] # parameters is a tuple   
        nD =  parameter[0]  
        nmax = parameter[1] 
        N = parameter[2] 
    #######################
    results_overN = []
    Ps = []
    for n in N:
        alpha = [0.75+0.25*i for i in range(0,10)]
        P = [int(a*n) for a in alpha]
        results = []
        epochs = []
        for p in P: # do experiment for each item in P
            suc_sum = 0
            dt_set = 0
            e = 0
            while dt_set < nD: 
                model = Perceptron(n,p)
                train, epoch = model.training(nmax)
                e += epoch 
                if train  == "success":   
                    suc_sum +=1
                dt_set += 1
            results.append(suc_sum / nD)
            epochs.append(e/nD)
        print("Results: " + str(results))
        results_overN += results
        if print_avgEpochs:
            print("Average number of Epochs for each P: "+ str(epochs))
        Ps += P
    plot_results(results_overN,alpha,P,N)

def only_plot():
    N = [10,20,100]
    alpha = [0.75+0.25*i for i in range(0,10)]
    P = []
    P = [int(a*N[0]) for a in alpha]
    # Insert result printouts
    results = [1.0, 1.0, 1.0, 0.84, 0.72, 0.46, 0.36, 0.22, 0.12, 0.06] + [1.0, 1.0, 1.0, 0.9, 0.52, 0.24, 0.12, 0.04, 0.02, 0.0] + [1.0, 1.0, 1.0, 0.9, 0.28, 0.0, 0.0, 0.0, 0.0, 0.0]
    plot_results(results,alpha,P,N)
    
    # copy paste output of previous results
    #Results: [1.0, 1.0, 1.0, 0.84, 0.72, 0.46, 0.36, 0.22, 0.12, 0.06]
    #Average number of Epochs for each P: [1.94, 4.2, 6.28, 27.72, 40.04, 70.54, 72.92, 86.88, 91.96, 96.24]
    #Results: [1.0, 1.0, 1.0, 0.9, 0.52, 0.24, 0.12, 0.04, 0.02, 0.0]
    #Average number of Epochs for each P: [3.54, 7.4, 11.24, 25.84, 65.22, 84.32, 96.36, 97.04, 99.28, 100.0]
    #Results: [1.0, 1.0, 1.0, 0.9, 0.28, 0.0, 0.0, 0.0, 0.0, 0.0]
    #Average number of Epochs for each P: [6.18, 10.78, 22.0, 51.72, 91.6, 100.0, 100.0, 100.0, 100.0, 100.0]

    #nD =  500 nmax = 1000 N = [10,20,100]
    #Results: [1.0, 1.0, 0.992, 0.938, 0.87, 0.59, 0.448, 0.214, 0.096, 0.058]
    #Average number of Epochs for each P: [1.842, 4.476, 19.88, 91.656, 167.626, 460.6, 616.088, 822.474, 915.564, 951.272]
    #Results: [1.0, 1.0, 1.0, 0.956, 0.78, 0.462, 0.19, 0.064, 0.014, 0.006]
    #Average number of Epochs for each P: [3.288, 6.128, 16.214, 89.0, 304.178, 618.432, 852.636, 954.318, 989.138, 995.21]


if __name__ == "__main__":
    if sys.argv[1] == "run":
        print("Running Experiment...")
        if len(sys.argv) >2:
            parameter  = ast.literal_eval(sys.argv[2])
            print('Input parameters are: '+ str(parameter))
            print_avgEpochs = sys.argv[3] == "True"
            if print_avgEpochs:
                print("Printing Epochs enabled.")
            else:
                print("Printing Epochs disabled.")
            print("=====================")
            run_experiment(print_avgEpochs,parameter)
        else:
            print("Utilizing standard parameters.")
            print_avgEpochs = True # if True will print out mean epochs to reach results for each P.
            run_experiment(print_avgEpochs) 
    elif sys.argv[1] == "plot":
        only_plot()