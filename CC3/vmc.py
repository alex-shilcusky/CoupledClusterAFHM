import numpy as np
import random
from CC3 import model 
from CC3 import wf 
from CC3 import bits 
from CC3 import tableaux

class CC(object):
    
    def __init__ (self,H,Hupdate,Nmax,tmax = 2):
       
        self.model = H
        self.update_model = Hupdate 
        self.N = H.N 
        self.L0 = H.L0
        self.state = np.zeros(self.N,dtype='int')
        self.weight = 1
        
        self.t_list = tableaux.generate_tableaux_list(Nmax,tmax)


def metropolis(CC,t1,t2,t3,niter,ndecorr,calc_derivs=False):
    res = []
    N = CC.model.N
    dim = CC.model.dim
    Ex = model.energy(CC.model,CC.state,CC.t_list,t1,t2,t3)
    aux_dpsi = np.zeros((N,N,dim,dim))
    aux_Hdpsi = np.zeros((N,N,dim,dim))
    aux_dpsi3 = np.zeros((N,N,N,dim,dim,dim))
    aux_Hdpsi3 = np.zeros((N,N,N,dim,dim,dim))

    dpsi = np.zeros((N,N,dim,dim))
    Hdpsi = np.zeros((N,N,dim,dim))
    dpsi3 = np.zeros((N,N,N,dim,dim,dim))
    Hdpsi3 = np.zeros((N,N,N,dim,dim,dim))

    for iter in range(niter):
        for iter_decorr in range(ndecorr):
            new_state = np.copy(CC.state)

            Hx = model.apply_H(CC.model,new_state,False)
#            Hx = model.apply_H_old(CC.model,new_state)
            (i,j,ni,nj,coef) = Hx[np.random.randint(len(Hx))]
            new_state[i] = ni
            new_state[j] = nj

#            found = False
#            while(not found):
#                (i,j,ni,nj,coef) = Hx[np.random.randint(len(Hx))]
#
#                new_state[i] = ni
#                new_state[j] = nj
#                if(new_state[i] != CC.state[i] or new_state[j] != CC.state[j]):
#                    found = True

            new_weight = wf.Psi_x(new_state,CC.t_list,t1,t2,t3)**2
            old_weight = CC.weight

            w = new_weight/old_weight

            if(w > 1 or random.random() < w):
                CC.state[:] = new_state[:]
                CC.weight = new_weight
                if(calc_derivs):
                    Ex,aux_Hdpsi,aux_Hdpsi3,aux_dpsi,aux_dpsi3 = model.HO(CC.model,CC.state,CC.t_list,t1,t2,t3)
                else:
                    Ex = model.energy(CC.model,CC.state,CC.t_list,t1,t2,t3)

        if(calc_derivs):            
            Hdpsi += aux_Hdpsi
            dpsi += aux_dpsi
            Hdpsi3 += aux_Hdpsi3
            dpsi3 += aux_dpsi3

        res += [Ex] #/CC.N/CC.L0] 

    return res,Hdpsi/niter,Hdpsi3/niter,dpsi/niter,dpsi3/niter


from CC3 import model_custom
from CC3 import wf_custom

def exact_sum(CC,basis,t1,t2,t3):

    N = CC.model.N
    dim = CC.model.dim
    dpsi = np.zeros((N,dim))
    Hdpsi = np.zeros((N,dim))
    dpsi2 = np.zeros((N,N,dim,dim))
    Hdpsi2 = np.zeros((N,N,dim,dim))
    dpsi3 = np.zeros((N,N,N,dim,dim,dim))
    Hdpsi3 = np.zeros((N,N,N,dim,dim,dim))

    aux_dpsi = np.zeros((N,dim))
    aux_Hdpsi = np.zeros((N,dim))
    aux_dpsi2 = np.zeros((N,N,dim,dim))
    aux_Hdpsi2 = np.zeros((N,N,dim,dim))
    aux_dpsi3 = np.zeros((N,N,N,dim,dim,dim))
    aux_Hdpsi3 = np.zeros((N,N,N,dim,dim,dim))

#    basis = CC.model.full_basis();

    norm = 0.
    E0 = 0.
    nstart = 0
    for state in basis[nstart:]:
        new_state = np.copy(state)

        Ex,aux_Hdpsi,aux_Hdpsi2,aux_Hdpsi3 , \
            aux_dpsi,aux_dpsi2, aux_dpsi3 = \
                model_custom.HO(CC.model,new_state,CC.t_list,t1,t2,t3)
        
        psi2 = wf_custom.Psi_x(new_state,CC.t_list,t1,t2,t3)**2
        norm += psi2

        E0 += Ex*psi2
        Hdpsi += aux_Hdpsi*psi2
        dpsi += aux_dpsi*psi2
        Hdpsi2 += aux_Hdpsi2*psi2
        dpsi2 += aux_dpsi2*psi2
        Hdpsi3 += aux_Hdpsi3*psi2
        dpsi3 += aux_dpsi3*psi2
   
#    print(norm) 
    E0 /= norm
    Hdpsi /= norm
    dpsi /= norm
    Hdpsi2 /= norm
    dpsi2 /= norm
    Hdpsi3 /= norm
    dpsi3 /= norm

    return E0,Hdpsi,Hdpsi2, Hdpsi3,dpsi,dpsi2,dpsi3

def exact_sum_coefficients(CC,t1,t2,t3):

    basis = CC.model.full_basis();
    psi2 = []

    norm = 0.
    for state in basis:
        psi2 += [wf.Psi_x(state,CC.t_list,t1,t2,t3)**2]
        norm += psi2[-1]

    norm_psi = []
    for x in psi2:
        norm_psi += [x/norm]
         
    return basis,norm_psi

def ED_matrix(CC):

    basis = CC.model.full_basis();
    Hmatrix = np.zeros((len(basis),len(basis)))

    for n1 in range(len(basis)):
        state = basis[n1]
        print(n1,state)
        Hx = model.apply_H(CC.model,state)
        for (i,j,ni,nj,coef) in Hx:
            new_state = np.copy(state)
            new_state[i] = ni
            new_state[j] = nj
            index = -1
            for n2 in range(len(basis)):
                if(np.all(basis[n2] == new_state)):
                    index = n2
                    break
            if(index == -1):
                print("ERROR: state not found ",state,new_state) 
            Hmatrix[n1,index] += coef
                
    return Hmatrix 

def generate_constrained_basis(CC,t1,t2,t3,nstates):
    ref_state = np.zeros(CC.N,dtype='int')
    aux_basis = [ref_state]  # this is |0>|0>...|0> (singlet coupled state)

    Hx = model.apply_H(CC.model,ref_state)

    for (i,j,ni,nj,coef) in Hx:
        aux_state = np.zeros(CC.N,dtype='int')
        aux_state[i] = ni
        aux_state[j] = nj

        found = False
        for state in aux_basis:
            if(np.all(state == aux_state)):
                found = True
                break
        if(not found):
            aux_basis += [aux_state]

    ntry = 0    
    ref_state = aux_basis[np.random.randint(len(aux_basis))]
    wold = wf.Psi_x(ref_state,CC.t_list,t1,t2,t3)**2
    while(len(aux_basis) < 5*nstates):
        new_state = aux_basis[np.random.randint(len(aux_basis))]
        wnew = wf.Psi_x(new_state,CC.t_list,t1,t2,t3)**2
        w = wnew/wold
        if(w < 1. and random.random() > w):
            continue

        wold = wnew
        ref_state = new_state
        Hx = model.apply_H(CC.model,ref_state)
        for (i,j,ni,nj,coef) in Hx:
            aux_state = np.copy(ref_state)
            aux_state[i] = ni
            aux_state[j] = nj
  
            found = False
            for state in aux_basis:
                if(np.all(state == aux_state)):
                    found = True
                    ntry += 1
                    break
            if(not found):
                aux_basis += [aux_state]
                ntry = 0
                
        if(ntry > 100000):
            break

    psi_list = []
    for state in aux_basis:
        psi_list += [wf.Psi_x(state,CC.t_list,t1,t2,t3)**2]
   
    print("Full DIM = ",len(aux_basis))
 
    sorted_indices = sorted(range(len(psi_list)), key=lambda i: psi_list[i], reverse=True)
    new_basis = []
    new_psi = []
    for i in range(min(nstates,len(aux_basis))):
        new_basis += [aux_basis[sorted_indices[i]]]
        new_psi += [psi_list[sorted_indices[i]]]

    print("Truncated DIM = ",len(new_basis))

    return new_basis, new_psi


# def generate_constrained_basis(CC,t1,t2,t3,nstates):
#     ref_state = np.zeros(CC.N,dtype='int')
#     aux_basis = [ref_state]
#     psi_list = []

#     Hx = model.apply_H(CC.model,ref_state)

#     for (i,j,ni,nj,coef) in Hx:
#         aux_state = np.zeros(CC.N,dtype='int')
#         aux_state[i] = ni
#         aux_state[j] = nj

#         found = False
#         for state in aux_basis:
#             if(np.all(state == aux_state)):
#                 found = True
#                 break
#         if(not found):
#             aux_basis += [aux_state]

#     ntry = 0    
#     while(len(aux_basis) < 5*nstates):
#         ref_state = aux_basis[np.random.randint(len(aux_basis))]
#         Hx = model.apply_H(CC.model,ref_state)
#         wold = wf.Psi_x(ref_state,CC.t_list,t1,t2,t3)**2
#         for (i,j,ni,nj,coef) in Hx:
#             aux_state = np.copy(ref_state)
#             aux_state[i] = ni
#             aux_state[j] = nj
  
#             found = False
#             for state in aux_basis:
#                 if(np.all(state == aux_state)):
#                     found = True
#                     ntry += 1
#                     break
#             if(not found):
#                 w = wf.Psi_x(state,CC.t_list,t1,t2,t3)**2/wold
#                 if(w >= 1. or random.random() < w):
#                     aux_basis += [aux_state]
#                     ntry = 0
                
#         if(ntry > 1000):
#             break

#     for state in aux_basis:
#         psi_list += [wf.Psi_x(state,CC.t_list,t1,t2,t3)**2]
   
#     print("Full DIM = ",len(aux_basis))
 
#     sorted_indices = sorted(range(len(psi_list)), key=lambda i: psi_list[i], reverse=True)
#     new_basis = []
#     new_psi = []
#     for i in range(min(nstates,len(aux_basis))):
#         new_basis += [aux_basis[sorted_indices[i]]]
#         new_psi += [psi_list[sorted_indices[i]]]

#     print("Truncated DIM = ",len(new_basis))

#     return new_basis, new_psi


# def generate_constrained_basis(CC,t1,t2,t3,nstates):
#     ref_state = np.zeros(CC.N,dtype='int')
#     aux_basis = [ref_state]
#     psi_list = []

#     Hx = model.apply_H(CC.model,ref_state)

#     for (i,j,ni,nj,coef) in Hx:
#         aux_state = np.zeros(CC.N,dtype='int')
#         aux_state[i] = ni
#         aux_state[j] = nj

#         found = False
#         for state in aux_basis:
#             if(np.all(state == aux_state)):
#                 found = True
#                 break
#         if(not found):
#             aux_basis += [aux_state]
    
#     while(len(aux_basis) < 5*nstates):
#         ref_state = aux_basis[np.random.randint(len(aux_basis))]
#         Hx = model.apply_H(CC.model,ref_state)
#         for (i,j,ni,nj,coef) in Hx:
#             aux_state = np.copy(ref_state)
#             aux_state[i] = ni
#             aux_state[j] = nj
  
#             found = False
#             for state in aux_basis:
#                 if(np.all(state == aux_state)):
#                     found = True
#                     break
#             if(not found):
#                 aux_basis += [aux_state]

#     for state in aux_basis:
#         psi_list += [wf.Psi_x(state,CC.t_list,t1,t2,t3)**2]
    
#     sorted_indices = sorted(range(len(psi_list)), key=lambda i: psi_list[i], reverse=True)
#     new_basis = []
#     new_psi = []
#     for i in range(nstates):
#         new_basis += [aux_basis[sorted_indices[i]]]
#         new_psi += [psi_list[sorted_indices[i]]]

#     return new_basis, new_psi

def metropolis_constrained(CC,basis,t1,t2,t3,niter,ndecorr,calc_derivs=False):
    ref_state = np.zeros(CC.N,dtype='int')
    Ex_list = []
    psi_list = []
    dpsi_list = []
    Hdpsi_list = []
    dpsi3_list = []
    Hdpsi3_list = []

    for state in basis:
        psi_list += [wf.Psi_x(state,CC.t_list,t1,t2,t3)**2]
        Ex,Hdpsi,Hdpsi3,dpsi,dpsi3 = model.HO(CC.model,state,CC.t_list,t1,t2,t3,level*2)
        Ex_list += [Ex]
        Hdpsi_list += [Hdpsi]
        dpsi_list += [dpsi]
        Hdpsi3_list += [Hdpsi3]
        dpsi3_list += [dpsi3]
    
    Ex = 0.
    N = CC.model.N
    dim = CC.model.dim
    dpsi = np.zeros((N,N,dim,dim))
    Hdpsi = np.zeros((N,N,dim,dim))
    dpsi3 = np.zeros((N,N,N,dim,dim,dim))
    Hdpsi3 = np.zeros((N,N,N,dim,dim,dim))
    aux_dpsi = np.zeros((N,N,dim,dim))
    aux_Hdpsi = np.zeros((N,N,dim,dim))
    aux_dpsi3 = np.zeros((N,N,N,dim,dim,dim))
    aux_Hdpsi3 = np.zeros((N,N,N,dim,dim,dim))

    old_state = 0
    new_state = 0
    CC.weight = 1.
    res = []
    for iter in range(niter):
        for iter_decorr in range(ndecorr):
            found = False
            while(not found):
                new_state = np.random.randint(len(basis))
                if(new_state != old_state):
                    found = True 

            new_weight = psi_list[new_state]
            old_weight = CC.weight

            w = new_weight/old_weight
#            print(iter,iter_decorr,old_state,new_state,w)

            if(w >= 1. or random.random() < w):
                old_state = new_state
                CC.weight = new_weight

        if(calc_derivs):            
            Hdpsi += Hdpsi_list[old_state]
            dpsi += dpsi_list[old_state]
            Hdpsi3 += Hdpsi3_list[old_state]
            dpsi3 += dpsi3_list[old_state]

        res += [Ex_list[old_state]] 

    CC.state = basis[old_state]
    return res,Hdpsi/niter,Hdpsi3/niter,dpsi/niter,dpsi3/niter


def metropolis_constrained_orig(CC,t1,t2,t3,niter,ndecorr,level=1,calc_derivs=False):
    ref_state = np.zeros(CC.N,dtype='int')
    aux_basis = [ref_state]
    Ex_list = []
    psi_list = []
    dpsi_list = []
    Hdpsi_list = []
    dpsi3_list = []
    Hdpsi3_list = []

    Hx = model.apply_H(CC.model,ref_state)

    for (i,j,ni,nj,coef) in Hx:
        aux_state = np.zeros(CC.N,dtype='int')
        aux_state[i] = ni
        aux_state[j] = nj

        found = False
        for state in aux_basis:
            if(np.all(state == aux_state)):
                found = True
                break
        if(not found):
            aux_basis += [aux_state]

    if(level > 1):
        for ref_state in aux_basis:
            Hx = model.apply_H(CC.model,ref_state)
            for (i,j,ni,nj,coef) in Hx:
                aux_state = np.copy(ref_state)
                aux_state[i] = ni
                aux_state[j] = nj
  
                found = False
                for state in aux_basis:
                    if(np.all(state == aux_state)):
                        found = True
                        break
                if(not found):
                    aux_basis += [aux_state]

    for state in aux_basis:
        psi_list += [wf.Psi_x(state,CC.t_list,t1,t2,t3)**2]
        Ex,Hdpsi,Hdpsi3,dpsi,dpsi3 = model.HO(CC.model,state,CC.t_list,t1,t2,t3,level*2)
        Ex_list += [Ex]
        Hdpsi_list += [Hdpsi]
        dpsi_list += [dpsi]
        Hdpsi3_list += [Hdpsi3]
        dpsi3_list += [dpsi3]
    
#    print(aux_basis)
#    print(Ex_list)
#    print(psi_list)

    Ex = 0.
    N = CC.model.N
    dim = CC.model.dim
    dpsi = np.zeros((N,N,dim,dim))
    Hdpsi = np.zeros((N,N,dim,dim))
    dpsi3 = np.zeros((N,N,N,dim,dim,dim))
    Hdpsi3 = np.zeros((N,N,N,dim,dim,dim))
    aux_dpsi = np.zeros((N,N,dim,dim))
    aux_Hdpsi = np.zeros((N,N,dim,dim))
    aux_dpsi3 = np.zeros((N,N,N,dim,dim,dim))
    aux_Hdpsi3 = np.zeros((N,N,N,dim,dim,dim))

    old_state = 0
    new_state = 0
    CC.weight = 1.
    res = []
    for iter in range(niter):
        for iter_decorr in range(ndecorr):
            found = False
            while(not found):
                new_state = np.random.randint(len(aux_basis))
                if(new_state != old_state):
                    found = True 

            new_weight = psi_list[new_state]
            old_weight = CC.weight

            w = new_weight/old_weight
#            print(iter,iter_decorr,old_state,new_state,w)

            if(w >= 1. or random.random() < w):
                old_state = new_state
                CC.weight = new_weight

        if(calc_derivs):            
            Hdpsi += Hdpsi_list[old_state]
            dpsi += dpsi_list[old_state]
            Hdpsi3 += Hdpsi3_list[old_state]
            dpsi3 += dpsi3_list[old_state]

        res += [Ex_list[old_state]] 

    CC.state = aux_basis[old_state]
    return res,Hdpsi/niter,Hdpsi3/niter,dpsi/niter,dpsi3/niter


