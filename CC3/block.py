import numpy as np
from CC3 import bits
from CC3 import lattice

# Using quantum number conservation : Sz

def build_basis(L,Nup):
    basis = []
    maxdim = (1 << L)-1
    for state in range(maxdim+1):
        n_ones = bits.count_bits(state,L)
        if(n_ones == Nup):
            basis += [state]
    return basis
        

#hflip[:] *= 2
# We build Hamiltonian matrix


# build operators and stores them into a list [H,Sz0,S+0,Sz1,S+1,....,Sz(L-1),S+(L-1),Parity,Sztot]
def calc_operators(basis, nn, sign_rule = False):
    dim = len(basis)
    L = nn.shape[0]
    H = np.zeros((dim,dim))
    ops = [H]

    hdiag = np.array([1,-1,-1,1])*0.25
    hflip = np.array([0,1,1,0])*0.5
    if(sign_rule):
        hflip *= -1

    for i in range(dim):
        state = basis[i]

        # Diagonal term
        for site_i in range(L):
            site_j = nn[site_i, lattice.Direction.RIGHT]
            if(site_j != -1):   # This would happen for open boundary conditions
                two_sites = bits.IBITS(state,site_i) | (bits.IBITS(state,site_j) << 1)
                value = hdiag[two_sites]
                H[i,i] += value


        # Off-diagonal term -(S+S-+S-S+) (we use the Marshall sign rule)
        for site_i in range(L):
            site_j = nn[site_i, lattice.Direction.RIGHT]

            if(site_j != -1):
                mask = (1 << site_i) | (1 << site_j)
                two_sites  = bits.IBITS(state,site_i) | (bits.IBITS(state,site_j) << 1)
                value = hflip[two_sites]
                if(value != 0.):
                    new_state = (state ^ mask) #XOR
#                    j = bisect(new_state, basis)
                    j = i
                    for k in range(dim):
                        if(new_state == basis[k]):
                            j = k
                    H[i,j] += value
#    print(H)              
    # Spin operators
    Sztot = np.zeros((dim,dim))
    for site_i in range(L):
        Sz = np.zeros((dim,dim))
        Splus = np.zeros((dim,dim))

        for i in range(dim):
            state = basis[i]
            Sz[i,i] = (2*bits.IBITS(state,site_i)-1)/2
            Sztot[i,i] += (2*bits.IBITS(state,site_i)-1)/2
            new_state = bits.IBSET(state, site_i)
            if(new_state != state):
#                j = bisect(new_state, basis)
                j = i
                for k in range(dim):
                    if(new_state == basis[k]):
                        j = k
                Splus[i,j] = 1.

        ops += [Sz]
        ops += [Splus]


    # Parity/reflection operator
    Parity = np.zeros((dim,dim))
    for i in range(dim):
        state = basis[i]
        new_state = 0
        for site_i in range(L):
            if(bits.IBITS(state,site_i) == 1):
                new_state = bits.IBSET(new_state,L-1-site_i)
        j = i
        if(new_state != state):
            for k in range(dim):
                if(new_state == basis[k]):
                    j = k
        Parity[i,j] = 1.

    ops += [Parity]
    ops += [Sztot]
    return ops

def rotate_operators(ops,subspaces):
    n = 0
    U = np.zeros(ops[0].shape)
    e = np.zeros(ops[0].shape[0])
    for dim in subspaces:
        H = ops[0][n:n+dim,n:n+dim]
        w, v = np.linalg.eigh(H)
        U[n:n+dim,n:n+dim] = v
        e[n:n+dim] = w
        print(w)
        n += dim
        
    idx = np.argsort(e)
    V = U[:,idx]

    new_ops = []
    for op in ops:
        new_op = V.T @ op @ V
        new_ops += [np.copy(new_op)]

    return new_ops

def cluster(L, sign_rule = False):
    bc = lattice.BoundaryCondition.OBC
    l = lattice.chain(L,bc)
    nn = l.nn
    basis = []
    subspaces = []
    for Nup in range(L+1):
        new_basis = build_basis(L, Nup)
        subspaces += [len(new_basis)]
        basis += new_basis
    ops = calc_operators(basis, nn, sign_rule)
    new_ops = rotate_operators(ops,subspaces)
    # cleanup
    for op in new_ops:
        for i in range(op.shape[0]):
            for j in range(op.shape[1]):
                if(abs(op[i][j]) < 1.e-10):
                    op[i][j] = 0.
    return basis, new_ops
