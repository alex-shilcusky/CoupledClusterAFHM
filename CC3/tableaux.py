import copy
def format_tableau(tableau):
    """
    Format a tableau for nice printing.
    
    Args:
        tableau: A 2D list representing a tableau
        
    Returns:
        A string representation of the tableau
    """
    result = []
    for row in tableau:
        result.append(" ".join(str(x) for x in row))
    return "\n".join(result)

def tableau_size(tableau):
    size = 0
    for row in tableau:
        size += len(row)
    return size 

def generate_all_tableaux(n,max_per_row=9999):
    all = []
    t = [[0]]
    all += [t]
    for i in range(1,n):
        new_to_add = []
        for t in all:
            for row in range(len(t)):
                tnew = copy.deepcopy(t)
                tnew[row] += [i]
                if(len(tnew[row]) <= max_per_row):
                    new_to_add += [tnew]
            tnew = copy.deepcopy(t)
            tnew += [[i]]
            new_to_add += [tnew]

        all += new_to_add
    return all

def generate_tableaux(n,max_per_row = 9999):
    new_t = []
    all_t = generate_all_tableaux(n, max_per_row)
    for t in all_t:
        if(tableau_size(t) == n):
            new_t += [t]
    return new_t

def generate_tableaux_list(nmax,max_per_row = 9999):
    all_t = []
    for n in range(1,nmax+1):
        all_t += [generate_tableaux(n,max_per_row)]
    return all_t

def sorted_tableaux(n, max_per_row = 9999):
    sorted_t = [[] for _ in range(n)]
    all_t = generate_all_tableaux(n, max_per_row)
    for t in all_t:
        k = tableau_size(t)
        sorted_t[k-1] += [t]
    return sorted_t
        
