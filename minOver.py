import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as col
import os



def generate_data(N,P):
    data = np.random.normal(loc = 0,scale = 1,size= (N,P))
    weights = np.random.normal(loc = 0,scale = 1,size= (1,N))
    labels = np.sign(np.dot(weights,data))
    return (data, weights, labels)

def Minover(weights, data,N,labels):
    old_stability = np.array([0 for i in range(0,np.shape(data)[1])])
    stability = np.array([1 for i in range(0,np.shape(data)[1])])
    
    while np.sum((old_stability-stability)**2) >0.5:
        Es = []
        for index in range(0,np.shape(data)[1]):
            E = np.dot(weights,data[:,index])*labels[0,index]
            Es.append(E)
        minE = np.argmin(Es)
        weights = weights + (data[:,minE]*labels[0,minE]/N)
        old_stability=stability
        stability = np.array(Es)/np.linalg.norm(weights)
    return weights, stability  

def GeneralizationError(weights, ideal_weights):
    ideal_weights = np.transpose(ideal_weights)
    return np.arccos(np.dot(weights,ideal_weights)/(np.linalg.norm(weights)*np.linalg.norm(ideal_weights)))/np.pi


def plot_results(results,alpha):
    fig = plt.figure()
    data_unified = {'Final Generalization Error':results,'P/N':alpha}
    data = pd.DataFrame(data_unified)
    sns.set(font_scale=1.3)
    sns.set_style("darkgrid") # ticks or whitegrid darkgrid
    # palette options: flare mako_r Set2 Set1 Set3
    sns.lineplot(data=data, x='P/N', y='Final Generalization Error',color="#cc0000")
    sns.scatterplot(data=data, x='P/N', y='Final Generalization Error',color="#cc0000")
    fig.savefig( "figures_2/errorplot.pdf")

def histograms(values,p):
    fig = plt.figure()
    data = {'Stability Values':values}
    sns.set(font_scale=1.3)
    sns.set_style("darkgrid") # ticks or whitegrid darkgrid
    
    sns.histplot(data=data,x = "Stability Values",color="#cc0000")
    fig.savefig( "figures_2/"+ str(p)+"histplot.pdf") 


def run_experiment():
    if not os.path.exists('figures_2'):
        os.makedirs('figures_2')
    # Need settings parameter again
    n = 100
    nD= 10 # number of experiments. min is 10 
    alpha = [0.25*i for i in range(1,28)]
    P = [int(a*n) for a in alpha]
    avg_errors = []
    for p in P: 
        avg_error= 0
        st_conc = np.array([])  
        for instance in range(0,nD):
            (data, ideal_weights, labels) = generate_data(n,p)
            weights = np.ones(n)
            weights *= 10**(-10) 
            weights, stability = Minover(weights,data,n,labels)
            if p in [150,350,525,675]:
                st_conc = np.concatenate((st_conc,stability), axis = 0)
            error = GeneralizationError(weights,ideal_weights)
            avg_error += error/nD
        if p in [150,350,525,675]:
            histograms(st_conc,p)
        avg_errors.append(avg_error[0])
    print(avg_errors)
    plot_results(avg_errors,alpha)

run_experiment()