import numpy as np
from math import comb, factorial
from CC3 import diagrams
from CC3 import tableaux 
from CC3 import model

def Psi_x(state,t_list,t1,t2,t3,echo=False):
    N = state.shape[0]
    t1_coef = []
    sites = []
    weight = 1.
    for i in range(N):
        if(state[i] != 0):
            t1_coef += [t1[0,state[i]]]
            sites += [i]
    nexcit = len(sites)    
#    print(nexcit)

    if(nexcit == 0):
        return weight
    
    t2_coef = np.zeros((nexcit,nexcit))
    t3_coef = np.zeros((nexcit,nexcit,nexcit))
    for i in range(nexcit):
        for j in range(i+1,nexcit):
            dist = (abs(sites[i]-sites[j]))
#            dist = dist - (2*dist//N)
            if(dist > N//2):
                dist = N-dist
#            print("t2 ",i,j,state[sites[i]],state[sites[j]],t2[sites[i],sites[j],state[sites[i]],state[sites[j]]])

          
#            t2_coef[i,j] = t2[0,abs(sites[i]-sites[j]),state[sites[i]],state[sites[j]]]    
#            t2_coef[i,j] = t2[0,abs(dist),state[sites[i]],state[sites[j]]]    
            t2_coef[i,j] = t2[sites[i],sites[j],state[sites[i]],state[sites[j]]]    

    if(nexcit < 3):
        weight = diagrams.coef_from_list(t_list[nexcit-1],t1_coef,t2_coef,t3_coef)
        return weight
    
    for i in range(nexcit):
        for j in range(i+1,nexcit):
            dij = (abs(sites[i]-sites[j]))
#            dij = dij - (2*dij//N)
            if(dij > N//2):
                dij = N-dij
            for k in range(j+1,nexcit):
                djk = (abs(sites[j]-sites[k]))
#                djk = djk - (2*djk//N)
                if(djk > N//2):
                    djk = N-djk

#                t3_coef[i,j,k] = t3[0,abs(dij),abs(djk),state[sites[i]],state[sites[j]],state[sites[k]]]    
                t3_coef[i,j,k] = t3[sites[i],sites[j],sites[k],state[sites[i]],state[sites[j]],state[sites[k]]]    

    weight = diagrams.coef_from_list(t_list[nexcit-1],t1_coef,t2_coef,t3_coef)

    return weight

def dPsi_x(state,t_list,t1,t2,t3,echo=False):
    N = state.shape[0]
    dim = t1.shape[1]
    dpsi = np.zeros((N,dim))
    dpsi_2 = np.zeros((N,N,dim,dim))
    dpsi_3 = np.zeros(t3.shape)
    t1_coef = []
    sites = []
    weight = 1.
    for i in range(N):
        if(state[i] != 0):
#            print("t1 ",state[i])
#            t1_coef += [t1[i,state[i]]]

            t1_coef += [t1[0,state[i]]]

            sites += [i]
    nexcit = len(sites)    

    if(nexcit == 0):
        return weight,dpsi,dpsi_2, dpsi_3
    
    t2_coef = np.zeros((nexcit,nexcit))
    t3_coef = np.zeros((nexcit,nexcit,nexcit))
    for i in range(nexcit):
        for j in range(i+1,nexcit):
            dist = (abs(sites[i]-sites[j]))
#            dist = dist - (2*dist//N)
            if(dist > N//2):
                dist = N-dist
#            t2_coef[i,j] = t2[0,abs(dist),state[sites[i]],state[sites[j]]]    
            t2_coef[i,j] = t2[sites[i],sites[j],state[sites[i]],state[sites[j]]]    

    if(nexcit < 3):
        weight,dpsi1,dpsi2,dpsi3 = diagrams.derivs_from_list(t_list[nexcit-1],t1_coef,t2_coef,t3_coef)
        for i in range(nexcit):
            # dpsi[0,0,state[sites[i]],state[sites[i]]] += dpsi1[i]
            dpsi[0,state[sites[i]]] += dpsi1[i]

            for j in range(nexcit):
                dist = (abs(sites[i]-sites[j]))
#                dist = dist - (2*dist//N)
                if(dist > N//2):
                    dist = N-dist
#                dpsi[0,abs(dist),state[sites[i]],state[sites[j]]] += dpsi2[i,j]
                dpsi_2[sites[i],sites[j],state[sites[i]],state[sites[j]]] += dpsi2[i,j]

        return weight,dpsi,dpsi_2, dpsi_3

    t3_coef = np.zeros((nexcit,nexcit,nexcit))
    for i in range(nexcit):
        for j in range(i+1,nexcit):
            dij = (abs(sites[i]-sites[j]))
#            dij = dij - (2*dij//N)
            if(dij > N//2):
                  dij = N-dij
            for k in range(j+1,nexcit):
                djk = (abs(sites[j]-sites[k]))
#                djk = djk - (2*djk//N)
                if(djk > N//2):
                    djk = N-djk

#                t3_coef[i,j,k] = t3[0,abs(dij),abs(djk),state[sites[i]],state[sites[j]],state[sites[k]]]    
                t3_coef[i,j,k] = t3[sites[i],sites[j],sites[k],state[sites[i]],state[sites[j]],state[sites[k]]]    

    weight,dpsi1,dpsi2,dpsi3 = diagrams.derivs_from_list(t_list[nexcit-1],t1_coef,t2_coef,t3_coef)

    for i in range(nexcit):
        for j in range(i+1,nexcit):
            dij = (abs(sites[i]-sites[j]))
#            dij = dij - (2*dij//N)
            if(dij > N//2):
                dij = N-dij
            for k in range(j+1,nexcit):
                djk = (abs(sites[j]-sites[k]))
#                djk = djk - (2*djk//N)
                if(djk > N//2):
                    djk = N-djk
#                dpsi_3[0,abs(dij),abs(djk),state[sites[i]],state[sites[j]],state[sites[k]]] += dpsi3[i,j,k]
                dpsi_3[sites[i],sites[j],sites[k],state[sites[i]],state[sites[j]],state[sites[k]]] += dpsi3[i,j,k]


    return weight,dpsi,dpsi_2,dpsi_3

