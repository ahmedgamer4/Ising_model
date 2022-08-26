from distutils.command.config import config
from tkinter import N
import numpy as np
import matplotlib as plt
from numpy.random import rand
from scipy.sparse import spdiags, linalg, eye

# Functions that will be used.

def initialstate(N):
    # Generate a random spin config for initial condition
    state = 2*np.random.randint(2, size=(N,N)) - 1
    return state


def mcmove(config, beta):
    for i in range(N):
        for j in range(N):
            a = np.random.randint(0, N)
            b = np.random.randint(0, N)
            s = config[a, b]
            nb = config[(a+1)%N,b] + config[a,(b+1)%N] + config[(a-1)%N,b] + config[a,(b-1)%N]
            cost = 2*s*nb
            
            if cost < 0:
                s += -1
            elif rand() < np.exp(-cost*beta):
                s += -1
            config[a, b] = s
        
    return config

def calc_energy(config):
    
    for i in range(len(config)):
        for j in range(len(config)):
            S = config[i, j]
            nb = config[(i+1)%N, j] + config[i,(j+1)%N] + config[(i-1)%N, j] + config[i,(j-1)%N]
            energy += -nb*S
    return energy / 2      


def calc_mag(config):
    # Magnetization of a given config
    
    mag = np.sum(config)
    return mag


N = 10              # Size of the lattice
n_t = 32            # Number of temperature points
eq_steps = 2**8     # Number of MC sweeps fo equilibration
mc_steps = 2**10    # Number of MC sweeps for calculation

T       = np.linspace(1.53, 3.28, nt); 
E,M,C,X = np.zeros(n_t), np.zeros(n_t), np.zeros(n_t), np.zeros(n_t)
n1, n2  = 1.0/(mc_steps*N*N), 1.0/(mc_steps*mc_steps*N*N) 