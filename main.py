import numpy as np
import matplotlib as plt
from numpy.random import rand
from scipy.sparse import spdiags, linalg, eye

# Functions that will be used.

N = 10              # Size of the lattice
n_t = 32            # Number of temperature points
eq_steps = 2**8     # Number of MC sweeps fo equilibration
mc_steps = 2**10    # Number of MC sweeps for calculation

T       = np.linspace(1.53, 3.28, n_t); 
E,M,C,X = np.zeros(n_t), np.zeros(n_t), np.zeros(n_t), np.zeros(n_t)
n1, n2  = 1.0/(mc_steps*N*N), 1.0/(mc_steps*mc_steps*N*N) 

def initialstate():
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
    energy = 0
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


# Main

for tt in range(n_t):
    config = initialstate()
    
    E1 = M1 = E2 = M2 = 0
    iT = 1.0/T[tt]
    iT2 = iT * iT
    
    for i in range(eq_steps):
        mcmove(config, iT)
        
    for i in range(mc_steps):
        mcmove(config, iT)
        ene = calc_energy(config)
        mag = calc_mag(config)
        
        E1 = E1 + ene
        M2 = M2 + mag
        E2 = E2 + ene*ene
        M1 = M1 + mag*mag
        
        
    E[tt] = n1*E1
    M[tt] = n1*M1
    C[tt] = (n1*E2 - n2*E1*E1)*iT2
    X[tt] = (n1*M2 - n2*M1*M1)*iT
    
f = plt.figure(figsize=(18, 10)); #  


sp =  f.add_subplot(2, 2, 1 )
plt.scatter(T, E, s=50, marker='o', color='IndianRed')
plt.xlabel("Temperature (T)", fontsize=20);
plt.ylabel("Energy ", fontsize=20)      
plt.axis('tight')


sp =  f.add_subplot(2, 2, 2 );
plt.scatter(T, abs(M), s=50, marker='o', color='RoyalBlue')
plt.xlabel("Temperature (T)", fontsize=20)
plt.ylabel("Magnetization ", fontsize=20)  
plt.axis('tight')


sp =  f.add_subplot(2, 2, 3 );
plt.scatter(T, C, s=50, marker='o', color='IndianRed')
plt.xlabel("Temperature (T)", fontsize=20)  
plt.ylabel("Specific Heat ", fontsize=20)
plt.axis('tight') 


sp =  f.add_subplot(2, 2, 4 )
plt.scatter(T, X, s=50, marker='o', color='RoyalBlue')
plt.xlabel("Temperature (T)", fontsize=20) 
plt.ylabel("Susceptibility", fontsize=20)
plt.axis('tight')