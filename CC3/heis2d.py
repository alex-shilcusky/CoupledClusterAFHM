import numpy as np
from CC3 import bits
from CC3 import lattice
from CC3 import block
from CC3 import wf

class Heis2D(object):

    def __init__ (self,my_lattice,sign,add_flips = False):
        self.N = my_lattice.N
        self.L0 = 4
        self.bc = my_lattice.bc
        self.bond_terms = []
        self.local_terms = []

        ################ CREATE OPERATORS AND ROTATE TO EIGENBASIS
        self.block = lattice.chain(self.L0, lattice.BoundaryCondition.PBC)
        basis = []
        subspaces = []
        for Nup in range(self.L0+1):
            new_basis = block.build_basis(self.L0, Nup)
            subspaces += [len(new_basis)]
            basis += new_basis
            print(self.L0,Nup,new_basis)

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

        self.neighbors = my_lattice
   
        pairs = [] 
        pairs += [[(1,0),(2,3)]]
        pairs += [[(2,1),(3,0)]]

        for site1 in range(self.N):
            for direction in [lattice.Direction.RIGHT,lattice.Direction.TOP]:
                site2 = self.neighbors.nn[site1,direction]
                if(site1 == site2):
                    continue
                 
                for (n1,n2) in pairs[direction]: 
                    print("BOND TERM: ",site1,site2,n1,n2)
                    sz1 = self.new_ops[2*n1+1]
                    sz2 = self.new_ops[2*n2+1]

                    sp1 = self.new_ops[2*n1+2]
                    sp2 = self.new_ops[2*n2+2]

                    Hbond = np.kron(sz1,sz2)
                    Hbond += np.kron(sp1,sp2.transpose())*(0.5*sign)
                    Hbond += np.kron(sp1.transpose(),sp2)*(0.5*sign)

                    self.bond_terms += [(site1,site2,np.copy(Hbond))]

        for i in range(self.N):
            self.local_terms += [(i,np.copy(self.new_ops[0]))]

        if(add_flips):
            Sztot = self.new_ops[-1].diagonal()
            dim = Sztot.shape[0]
            flip_op = np.zeros((dim,dim))
            for ni in range(dim):
                 for nj in range(ni+1,dim):
                     if(abs(Sztot[ni] - Sztot[nj]) < 1.e-8):
                         flip_op[ni,nj] = 1.
                         flip_op[nj,ni] = 1.
            for i in range(self.N):
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
