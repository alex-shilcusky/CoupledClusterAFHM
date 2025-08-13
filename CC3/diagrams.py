import numpy as np
from itertools import permutations, combinations
from math import comb, factorial
from CC3 import bits

########### generate all possible configurations of nbits ones in n bits
def gen_configs_ones(n, nbits):
    b = [(1 << nbits)-1]

    i=0
    j=0
    nmax=1
    while(i <= n-2):
        if(bits.IBITS(b[-1],i) == 1):
            if(bits.IBITS(b[-1], i+1) == 0):
                bold = b[-1]
                print(nmax,bold)
                nmax += 1
                b += [0]
                b[-1] = bits.IBCLR(b[-1],i)
                b[-1] = bits.IBSET(b[-1],i+1)
                for l in range(j):
                    b[-1] = bits.IBSET(b[-1],l)
                for l in range(j,i):
                    b[-1] = bits.IBCLR(b[-1],l)
                if(i <= n-2):
                    for l in range(i+2,i+2+n-i-1):
                        b[-1] = bits.IBCLR(b[-1],l)
                        if(bits.IBITS(bold,l) == 1):
                            bits.IBSET(b[-1],l)
#                    call bits.MVBITS(b(nmax-1), i+2, nsites-1-i-1, b(nmax), i+2)
                j=-1
                i=-1
            j=j+1
        i=i+1
    return b

def gen_configs(n, nbits):
    b = []
    for i in range((1 << n)-1):
        if(bits.count_bits(i,n) == nbits):
            b += [i]
    return b

########### generate all possible configurations of 2 ones in n bits that do not overlap with ref
def gen_configs2(n,ref):
    b = []
    for i in range((1 << n)):
        if((i&ref) == 0 and bits.count_bits(i,n) == 2):
            b += [i]
    return b    


########### generate all possible configurations of 2 ones in the (l,m) digram 
def gen_lm(l,m):

    b = []
    baux = gen_configs2(l,0) # first row
    for x in baux:
        b += [[x]] 

    nrows = l-m
       
    for row in range(1,nrows):
        bprev = b.copy()
        b.clear()
        for n in range(len(bprev)):
            ref = 0
            for x in bprev[n]:
                ref |= x
                
            baux = gen_configs2(l,ref)
            for x in baux:
                b += [bprev[n]+[x]]

    b_unique = []
    for config in b:
        config.sort()
        found = False
        for c2 in b_unique:
            nequal = 0
            for i in range(len(config)):
                if(c2[i] == config[i]):
                    nequal += 1
            if(nequal == len(config)):
                found = True
                break  
        if(not found):
            b_unique += [config]

#    return b
    return b_unique

# returns indices corresponding to pairs of ones
def prod_t2_list(x,l):
    res = []
    for xpair in x:
        ij = []
        for bit in range(l):
            if(bits.IBITS(xpair,bit) == 1):
                ij += [bit]
        res += [[ij[0],ij[1]]]
    return res       

# returns indices corresponding to single ones
def prod_t1_list(x,l):
    res = []
    mask_t1 = 0 
    for xpair in x:
        mask_t1 |= xpair 
            
    mask_t1 ^= ((1 << l)-1)
    if(mask_t1 != 0):
        for bit in range(l):
            if(bits.IBITS(mask_t1,bit) == 1):
                res += [[bit]] 
    return res

# returns product of t2's correspondong to a bit sequence of pairs
def prod_t2(t1,t2,x,l):
    res = 1.
    for xpair in x:
        ij = []
        for bit in range(l):
            if(bits.IBITS(xpair,bit) == 1):
                ij += [bit]
        res *= t2[ij[0],ij[1]]
    return res           

# returns product of t1's correspondong to a bit sequence of single ones
def prod_t1(t1,t2,x,l):
    res = 1.
    mask_t1 = 0 
    for xpair in x:
        mask_t1 |= xpair 
            
    mask_t1 ^= ((1 << l)-1)
    if(mask_t1 != 0):
        res = 1.
        for bit in range(l):
            if(bits.IBITS(mask_t1,bit) == 1):
                res *= t1[bit]                
    return res

# returns the coefficient corresponding to a term (l,m)
def coef_lm(t1,t2,l,m):
    coef = 0.
    
    if(l == m):
        coef = np.prod(t1)
    else:
        b = gen_lm(l,m)
        for x in b:
            coef += prod_t1(t1,t2,x,l)*prod_t2(t1,t2,x,l)
        
    t1_power = 2*m-l
    t2_power = l-m

#    coef /= factorial(l-m)
#    coef *= factorial(m)/factorial(l-m)
#    coef /= factorial(m)  # From the exponential of the T1+T2        

#    coef /= factorial(t1_power)
#    coef /= factorial(t2_power)
#    coef *= factorial(2*m-l)
#    coef *= factorial(l-m)

    return coef

# Given (l,m) generate all possible diagrams for T1 and T2
def coef_list(l,m):
    t_list = []
    
    if(l == m):
        t1_list = []
        for n in np.arange(l,dtype='int'):
            t1_list += [[n]]
        t_list += [t1_list]
    else:
        b = gen_lm(l,m)
        for x in b:
            t1_list = prod_t1_list(x,l)
            t2_list = prod_t2_list(x,l)
            t_list += [t2_list+t1_list]

    return t_list

# sum all contributions correspondong to a spin configuration and return the coefficient
def coef_from_list(terms,t1,t2,t3):
    coef = 0.
    for t in terms:
        this_coef = 1.
        for row in t:
            match len(row) :
                case 1:
                    this_coef *= t1[row[0]]
                case 2:
                    this_coef *= t2[row[0],row[1]]
                case 3:
                    this_coef *= t3[row[0],row[1],row[2]]
        
        coef += this_coef
        
    return coef    


# derivatives of a coefficient correspondong to a spin configuration
def derivs_from_list(terms,t1,t2,t3):
    dpsi3 = np.zeros(t3.shape)
    dpsi2 = np.zeros(t2.shape)
    dpsi1 = np.zeros(len(t1))
    coef = 0.
    for t in terms:
        this_coef = 1.
        for row in t:
            row_coef = 1.e-8
            match len(row):
                case 1:
                    row_coef = t1[row[0]]
                case 2:
                    row_coef = t2[row[0],row[1]]
                case 3:
                    row_coef = t3[row[0],row[1],row[2]]

            if(abs(row_coef) > 1.e-8):
                this_coef *= row_coef
            else: 
                this_coef *= 1.e-8

        for row in t:
            match len(row):
                case 1:
                    if(abs(t1[row[0]]) > 1.e-8):
                        dpsi1[row[0]] += this_coef/t1[row[0]]
                    else:
                        dpsi1[row[0]] += this_coef/1.e-8
                case 2:
                    if(abs(t2[row[0],row[1]]) > 1.e-8):
                        dpsi2[row[0],row[1]] += this_coef/t2[row[0],row[1]]
                    else:
                        dpsi2[row[0],row[1]] += this_coef/1.e-8
                case 3:
                    if(abs(t3[row[0],row[1],row[2]]) > 1.e-8):
                        dpsi3[row[0],row[1],row[2]] += this_coef/t3[row[0],row[1],row[2]]
                    else:
                        dpsi3[row[0],row[1],row[2]] += this_coef/1.e-8

        coef += this_coef
#    print(terms,dpsi1)
        
    return coef,dpsi1,dpsi2,dpsi3    

######################## THIS FUNCTION BELOW IS A TEST AND DOESN'T WORK
def gen_lm_unique(l,m):
    configs = []

    ones = gen_configs_ones(l,2*m-l)
    for c in ones:
        singles = []
        indices = []
        for i in range(l):
            if(bits.IBITS(c,i) == 0):
                indices += [i]
            else:
                singles += [i]

        pairs = combinations(indices,2)
        configs += [singles+list(pairs)]


    return configs
                  


