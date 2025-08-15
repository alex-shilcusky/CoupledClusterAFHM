import numpy as np
from CC3 import bits
from CC3 import lattice
from CC3 import block
from CC3 import wf

class Heisenberg(object):

    def __init__ (self,N,L0,bc,sign,add_flips = False):
        self.N = N
        self.L0 = L0
        self.bc = bc
        self.bond_terms = []
        self.local_terms = []

        ################ CREATE OPERATORS AND ROTATE TO EIGENBASIS
        self.block = lattice.chain(L0, lattice.BoundaryCondition.OBC)
        basis = []
        subspaces = []
        for Nup in range(L0+1):
            new_basis = block.build_basis(L0, Nup)
            subspaces += [len(new_basis)]
            basis += new_basis
            print(L0,Nup,new_basis)

        self.dim = len(basis)
        print("Dim=",self.dim)
        print("Basis:")
        print(basis)
        print(subspaces)
 
        self.ops = block.calc_operators(basis, self.block.nn, sign == -1)
        self.new_ops = block.rotate_operators(self.ops,subspaces)
        gs = 0
        e0 = 1000.
        for i in range(self.new_ops[0].shape[0]):
            print(i," Sz= ",self.new_ops[-1][i,i]," E= ",self.new_ops[0][i,i])
            if(self.new_ops[0][i,i] < e0):
                e0 = self.new_ops[0][i,i]
        print("Ground state: ",gs,e0)

        szL = self.new_ops[1]
        szR = self.new_ops[2*L0-1]

        spL = self.new_ops[2]
        spR = self.new_ops[2*L0]

        Hbond = np.kron(szR,szL)
        Hbond += np.kron(spR,spL.transpose())*(0.5*sign)
        Hbond += np.kron(spR.transpose(),spL)*(0.5*sign)
        xmax = N
        if(bc == lattice.BoundaryCondition.OBC):
            xmax = N-1
        for i in range(xmax): 
            j = (i+1)%N
            self.bond_terms += [(i,j,np.copy(Hbond))]

        for i in range(N): 
            self.local_terms += [(i,np.copy(self.new_ops[0]))]

#        for xi in range(self.dim):
#            for xj in range(self.dim):
#                for ni in range(self.dim):
#                    for nj in range(self.dim):
#                        coef1 = Hbond[xi*self.dim+xj,ni*self.dim+nj]
#                        coef2 = szR[xi,ni]*szL[xj,nj]
#                        coef2 += spR[xi,ni]*spL[nj,xj]*(0.5*sign)
#                        coef2 += spR[ni,xi]*spL[xj,nj]*(0.5*sign)
#                        if(abs(coef1-coef2) > 1.e-5):
#                            print("ERROR ",xi,xj,ni,nj,coef1,coef2)

    
        if(add_flips):
            Sztot = self.new_ops[-1].diagonal()
            dim = Sztot.shape[0]
            flip_op = np.zeros((dim,dim))
            for ni in range(dim):
                 for nj in range(ni+1,dim):
                     if(abs(Sztot[ni] - Sztot[nj]) < 1.e-8):
                         flip_op[ni,nj] = 1.
                         flip_op[nj,ni] = 1.
            
            for i in range(N): 
                self.local_terms += [(i,np.copy(flip_op))]

    def full_basis(self):
        basis = []
        Szop = self.new_ops[-1]
        for i in range(self.dim**self.N):
            state = bits.numberToBase(i,self.dim)
            sztot = 0.
            for d in state:
                sztot += Szop[d,d]
            if(abs(sztot) < 1.e-7):
                 new_state = np.zeros(self.N,dtype=int);
                 for bit in range(len(state)):
                     new_state[len(state)-bit-1] = state[bit]
#                 print(i,state,new_state)
                 basis += [new_state]
        return basis


                      
def apply_H(H,state,diagonal=True):
    dim = H.dim

    Hx = []
    for (i,term) in H.local_terms:
        xi = state[i]
        for ni in range(dim):
            if(ni == xi and (not diagonal)):
                continue
            coef = term[xi,ni]
            if(abs(coef) > 1.e-10):
                Hx += [(i,i,xi,ni,coef)]

    for (i,j,term) in H.bond_terms:
        xi = state[i]
        xj = state[j]

        for ni in range(dim):
            for nj in range(dim):
                if(ni == xi and nj == xj and (not diagonal)):
                    continue
                coef = term[xi*dim+xj,ni*dim+nj]
                if(abs(coef) > 1.e-10):
                    Hx += [(i,j,ni,nj,coef)]
    
    return Hx  #, H

def HO(H,state,t_list,t1,t2,t3,nmax=-1):
    N = H.N
    dim = H.dim
    # Hdpsi = np.zeros((N,N,dim,dim))
    Hdpsi = np.zeros((N,dim))
    Hdpsi2 = np.zeros((N,N,dim,dim))
    Hdpsi3 = np.zeros((N,N,N,dim,dim,dim))

    Hx = apply_H(H,state)
    psi,dpsi,dpsi2, dpsi3 = wf.dPsi_x(state,t_list,t1,t2,t3)

    dpsi /= psi
    dpsi2 /= psi
    dpsi3 /= psi

    E = 0.
    new_state = np.copy(state)
    if(nmax == -1):
        for (i,j,ni,nj,coef) in Hx:
            new_state[:] = state[:]
            new_state[i] = ni
            new_state[j] = nj

            psix = wf.Psi_x(new_state,t_list,t1,t2,t3)
            E += coef*psix
    else:
        for (i,j,ni,nj,coef) in Hx:
            new_state[:] = state[:]
            new_state[i] = ni
            new_state[j] = nj
            nexcit = np.count_nonzero(new_state)
            if(nexcit <= nmax):
                psix = wf.Psi_x(new_state,t_list,t1,t2,t3)
                E += coef*psix

    E /= psi
    Hdpsi[:,:] = E*dpsi[:,:]
    Hdpsi2[:,:,:,:] = E*dpsi2[:,:,:,:]
    Hdpsi3[:,:,:,:,:,:] = E*dpsi3[:,:,:,:,:,:]

    return E,Hdpsi,Hdpsi2, Hdpsi3,dpsi,dpsi2,dpsi3


def energy(H,state,t_list,t1,t2,t3):
    Hx = apply_H(H,state)

    E = 0.
    new_state = np.copy(state)
    for (i,j,ni,nj,coef) in Hx:
        new_state[:] = state[:]
        new_state[i] = ni
        new_state[j] = nj
        E += wf.Psi_x(new_state,t_list,t1,t2,t3)*coef
    E /= wf.Psi_x(state,t_list,t1,t2,t3)

    return E

def apply_H_old(H,state):
    dim = H.dim
    L = H.N
    Hx = []
    Sztot = H.new_ops[-1]
    szL = H.new_ops[1]
    szR = H.new_ops[2*H.L0-1]

    spL = H.new_ops[2]
    spR = H.new_ops[2*H.L0]

    for i in range(L):
        j = (i+1)%L
        xi = state[i]
        xj = state[j]
        Hx += [(i,i,xi,xi,H.new_ops[0][xi,xi])]
        for ni in range(dim):
            for nj in range(dim):
                coef = szR[xi,ni]*szL[xj,nj]
                if(abs(coef) > 1.e-10):
                    Hx += [(i,j,ni,nj,coef)]
        for ni in range(dim):
            for nj in range(dim):
                coef = -spL[xi,ni]*spR[nj,xj]*0.5

                if(abs(coef) > 1.e-10):
                    Hx += [(i,j,ni,nj,coef)]
                coef = -spL[ni,xi]*spR[xj,nj]*0.5
                if(abs(coef) > 1.e-10):
                    Hx += [(i,j,ni,nj,coef)]

    
    Hx_unique = []
    for (i1,j1,n1,m1,coef1) in Hx:
        found = False
        for n in range(len(Hx_unique)):
            (i2,j2,n2,m2,coef2) = Hx_unique[n]
            if(i1 == i2 and j1 == j2 and n1 == n2 and m1 == m2):                
                found = True
                Hx_unique[n] = (i2,j2,n2,m2,coef1+coef2)
                break
            if(i1 == j2 and j1 == i2 and n1 == m2 and m1 == n2):                
                found = True
                Hx_unique[n] = (i2,j2,n2,m2,coef1+coef2)
                break
        if(not found):
            Hx_unique += [(i1,j1,n1,m1,coef1)]
            
    Hx.clear()
    for (i1,j1,n1,m1,coef1) in Hx_unique:
        if(abs(coef1) > 1.e-10):
            Hx += [(i1,j1,n1,m1,coef1)]
    
    return Hx  #, H
    
def apply_block_flips(L0,state,ops):
    dim = ops[0].shape[0]
    Hx = []
    Sztot = ops[-1]
    L = state.shape[0]

    for i in range(L):
        xi = state[i]
        Szref = Sztot[xi,xi]
        for ni in range(dim):
            coef = Sztot[ni,ni]
            if(ni != xi and abs(coef-Szref) < 1.e-10):
                Hx += [(i,i,ni,ni,1.)]
    return Hx
