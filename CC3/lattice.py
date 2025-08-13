import numpy as np

class BoundaryCondition:
    OBC, PBC = range(2)

class Direction:
    RIGHT, TOP, LEFT, BOTTOM = range(4)

# Lattice geometry  (1D chain)

class chain(object):

    def __init__ (self, L, bc):
        self.L = L
        self.N = L
        self.bc = bc
        self.x = np.zeros(self.N, dtype=np.int16)
        self.nn = np.zeros(shape=(L,4), dtype=np.int16)
        for i in range(L):
            self.x[i] = i
            self.nn[i, Direction.RIGHT] = i-1
            self.nn[i, Direction.LEFT] = i+1

        if(bc == BoundaryCondition.OBC):   # Open Boundary Conditions
            self.nn[0, Direction.RIGHT] = -1    # This means error
            self.nn[L-1, Direction.LEFT] = -1
        else:                              # Periodic Boundary Conditions
            self.nn[0, Direction.RIGHT] = L-1   # We close the ring
            self.nn[L-1, Direction.LEFT] = 0



class square(object):
    
    def __init__ (self, L, bc):
      
        self.L = L
        self.N = L*L
        self.bc = bc
        
        # Initialize site positions
        # Initialize neighbors table for boundary conditions
        self.nn = np.zeros(shape=(self.N,4), dtype=np.int16)
        self.position = np.zeros(shape=(L,L), dtype=np.int16)
        self.x = np.zeros(self.N, dtype=np.int16)
        self.y = np.zeros(self.N, dtype=np.int16)

        # Periodic boundary conditions
        n = 0
        for iy in range(L):
            for ix in range(L):
                self.position[iy,ix] = n
                self.x[n] = ix
                self.y[n] = iy
                self.nn[n,Direction.LEFT] = n-1
                self.nn[n,Direction.RIGHT] = n+1
                self.nn[n,Direction.TOP] = n+L
                self.nn[n,Direction.BOTTOM] = n-L
                if(ix == 0):
                    self.nn[n,Direction.LEFT] = n+L-1
                    if(bc == BoundaryCondition.OBC):
                        self.nn[n,Direction.LEFT] = n
                if(ix == L-1):
                    self.nn[n,Direction.RIGHT] = n-(L-1)
                    if(bc == BoundaryCondition.OBC):
                        self.nn[n,Direction.RIGHT] = n
                if(iy == 0):
                    self.nn[n, Direction.BOTTOM] = n+(L-1)*L
                    if(bc == BoundaryCondition.OBC):
                        self.nn[n,Direction.BOTTOM] = n
                if(iy == L-1):
                    self.nn[n, Direction.TOP] = n-(L-1)*L
                    if(bc == BoundaryCondition.OBC):
                        self.nn[n,Direction.TOP] = n
                n += 1


class ladder(object):
    
    def __init__ (self, Lx, Ly, bc):
      
        self.Lx = Lx
        self.Ly = Ly
        self.N = Lx*Ly
        self.bc = bc
        
        # Initialize site positions
        # Initialize neighbors table for boundary conditions
        self.nn = np.zeros(shape=(self.N,4), dtype=np.int16)
        self.position = np.zeros(shape=(Ly,Lx), dtype=np.int16)
        self.x = np.zeros(self.N, dtype=np.int16)
        self.y = np.zeros(self.N, dtype=np.int16)

        # Periodic boundary conditions
        n = 0
        for iy in range(Ly):
            for ix in range(Lx):
                self.position[iy,ix] = n
                self.x[n] = ix
                self.y[n] = iy
                self.nn[n,Direction.LEFT] = n-1
                self.nn[n,Direction.RIGHT] = n+1
                self.nn[n,Direction.TOP] = n+Lx
                self.nn[n,Direction.BOTTOM] = n-Lx
                if(ix == 0):
                    self.nn[n,Direction.LEFT] = n+Lx-1
                    if(bc == BoundaryCondition.OBC or Lx == 1):
                        self.nn[n,Direction.LEFT] = n
                if(ix == Lx-1):
                    self.nn[n,Direction.RIGHT] = n-(Lx-1)
                    if(bc == BoundaryCondition.OBC or Lx == 1):
                        self.nn[n,Direction.RIGHT] = n
                if(iy == 0):
                    self.nn[n, Direction.BOTTOM] = n+(Ly-1)*Lx
                    if(bc == BoundaryCondition.OBC or Ly == 1):
                        self.nn[n,Direction.BOTTOM] = n
                if(iy == Ly-1):
                    self.nn[n, Direction.TOP] = n-(Ly-1)*Lx
                    if(bc == BoundaryCondition.OBC or Ly == 1):
                        self.nn[n,Direction.TOP] = n
                n += 1



