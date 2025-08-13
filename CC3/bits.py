def IBITS(n,i):
    return ((n >> i) & 1)

def IBSET(i,pos):
    return (i|(1 << pos))

def IBCLR(i,pos):
    b = i
    if(IBITS(i,pos) == 1):
        return (i^(1 << pos))
    return b

def count_bits(n,L):
    nbits=0
    for bit in range(L):
        nbits += IBITS(n,bit)
    return nbits

def numberToBase(n, b):
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return digits[::-1]
