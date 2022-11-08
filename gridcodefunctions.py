#!/usr/bin/env python
# coding: utf-8

# # GKP surface code revisited

# #### imports and general definitions

# In[1]:


import numpy as np
import numpy.linalg as la
from matplotlib import pyplot as plt
from abelian import * # for Hermite normal form  and linear algebra over integers
from sympy import Matrix, Integer
from scipy.special import gamma



# In[2]:


def Jsym(n): # symplectic form for n modes
    return np.block([[np.zeros((n,n)), np.eye(n)],[-np.eye(n),np.zeros((n,n))]])


def binary_ar(N, ones):
    temp = np.zeros(N)
    temp[np.asarray(ones)] = 1

    return temp

# In[3]:


np.set_printoptions(precision=3, linewidth=1000) # for concise output without wrapping


# # Square surface code (rotated)

#################################################################
# #### A useful surface code class
#################################################################

class surfaceCode(object):
    '''a container for surface code related functions'''
    def __init__(self, l, signed=True): # initialize code of distance 
        #self.qposx, self.qposy = np.meshgrid(np.linspace(0, d-1, d) , np.linspace(0, d-1, d), sparse=False)
        self.l = l
        self.N=l**2
        self.qposx = np.linspace(0, l-1, l) 
        self.qposy = np.linspace(0, l-1, l) 
        self.qubits = [(x,y) for x in self.qposx for y in self.qposy]
        self.signed=signed
    
    
    def HX(self):
        HX = []
        for y in range(self.l-1): # first take care of "bulk" stabilizers
            for x in range(self.l-1):
                if (self.l*x+y)%2 == 0:    # <=> pivot elem of checkerboard (x%2==0) and (y%2 ==0) for odd l.  Qubit index I(x,y)=ly+x
                    s = [0 for j in range(self.l**2)]
                    s[self.l*y + x] = (-1)**(self.signed)
                    s[self.l*y + x +1] = 1
                    s[self.l*(y+1) + x] = 1
                    s[self.l*(y+1) + x + 1] = (-1)**(self.signed)
                    HX.append(s)
                    
        for x in range(1,self.l-1,2): # stabilizers on top edge
            s = [0 for j in range(self.l**2)]
            s[x] = 1
            s[x + 1] = (-1)**(self.signed)
            HX.append(s)
            
        for x in range(0,self.l-2,2): # stabilizers on bottom edge
            s = [0 for j in range(self.l**2)]
            s[self.l*(self.l-1)+x] = (-1)**(self.signed)
            s[self.l*(self.l-1)+x + 1] = 1
            HX.append(s)
        
        return(np.array(HX))
        
    def HZ(self):
        HZ = []
        for y in range(self.l-1): # first take care of "bulk" stabilizers
            for x in range(self.l-1):
                if (self.l*x+y)%2 == 1: #<=> (x%2==1)^(y%2==1)
                    s = [0 for j in range(self.l**2)]
                    s[self.l*y + x] = 1
                    s[self.l*y + x +1] = 1
                    s[self.l*(y+1) + x] = 1
                    s[self.l*(y+1) + x + 1] = 1
                    HZ.append(s)
                    
        for y in range(0,self.l-2,2): # stabilizers on left edge
            s = [0 for j in range(self.l**2)]
            s[self.l*y] = 1
            s[self.l*(y+1)] = 1
            HZ.append(s)
            
        for y in range(1,self.l-1,2): # stabilizers on right edge
            s = [0 for j in range(self.l**2)]
            s[self.l*y + self.l-1] = 1
            s[self.l*(y+1) + self.l-1] = 1
            HZ.append(s)

        return(np.array(HZ))

    def Z_L_symp(self):
        z = np.append(np.ones(self.l), np.zeros(self.l ** 2 - self.l))
        return np.append(np.zeros(self.l ** 2), z)

    def X_L_symp(self):
        x = np.zeros(self.l ** 2)
        x[np.arange(0, self.l ** 2, self.l)] = 1
        return np.append(x, np.zeros(self.l ** 2))

    def logicals_symp(self):
        return np.stack([self.X_L_symp(), self.Z_L_symp()])

    def MGKP_red(self):
        # reduced single-mode GKP generators
        single_GKP = np.asarray([binary_ar(2 * self.N, (self.N * (i % 2) + i)) for i in range(self.N)])

        L_ = self.l - 1
        mid_index = int(np.round(L_ / 2) * (1 + self.l))
        single_GKP = np.vstack([single_GKP, binary_ar(2 * self.N, (self.N + mid_index))])

        return single_GKP

# # Square surface code (rotated)

#################################################################
# #### A useful XZZX code class
#################################################################

class XZZZCode(object):
    '''a container for XZZX code related functions'''
    def __init__(self, l, w, signed=True): # initialize code of dimensions l,w
        #self.qposx, self.qposy = np.meshgrid(np.linspace(0, d-1, d) , np.linspace(0, d-1, d), sparse=False)
        pass   
    
    
#################################################################
# # Computing the distance
#################################################################

def vol_Nsphere(n,r):
    return np.pi**(n/2)/gamma(n/2+1)*r**(n)



def is_orthogonal(B):
    BBT=B@B.T
    BBT_OD=BBT-np.diag(np.diag(BBT))
    return np.linalg.norm(BBT_OD, 'fro')<=1e-10

def shortest_dual_vector(L):
    
    J = Jsym(L**2)
    C = surfaceCode(L, signed=True) # instantiate auxiliary SC class
    #print(C.qubits, "\n")
    HX = C.HX() # X type parity check matrix
    HZ = C.HZ() # Z type parity check matrix
    MSurft = np.sqrt(np.pi)*np.block([[HX,np.zeros(np.shape(HX))],[np.zeros(np.shape(HX)),HZ]]) # corresponding generator matrix
    MSurf = (1./np.sqrt(2*np.pi)) * MSurft # rescaled generator matrix
    
    MGKPt = 2*np.sqrt(np.pi)*np.eye(2*L**2) # tilda generator matrix as in the notes --> rows are nullifiers
    MGKP = (1./np.sqrt(2*np.pi))*MGKPt # rescaled generator matrix --> the corresponding A matrix below is integer
    AGKP = MGKP@J@np.transpose(MGKP)
    
    Mtot = np.vstack([MGKP,MSurf])
    
    Lmat = np.sqrt(2)*Mtot # make the matrix integer to work with lattice functions

    Lmat = Matrix(Lmat.astype(int))
    
    U,H = hermite_normal_form(Lmat.T) # U is unimodular, the columns of H correspond to the new nullifiers


    # let's convert back to numpy float arrays
    U = np.array(U).astype(np.float64)
    H = np.array(H).astype(np.float64)

    newMmat = (1/np.sqrt(2))*H.T # this should be the new generator matrix, rows are the new independent generators
    gen = newMmat[~np.all(newMmat == 0, axis=1)]
    
    gen_perp=np.linalg.inv(J@gen.T)
    
    gen_perp_mod=gen_perp-np.round(gen_perp@np.linalg.inv(gen))@gen
    gen_perp_mod[np.abs(gen_perp_mod)<1e-7]=0
    gen_perp_mod=gen_perp_mod[~np.all(gen_perp_mod == 0, axis=1)]*np.sqrt(2*np.pi)#/np.sqrt(np.pi)
    norms=np.asarray([np.linalg.norm(x,2) for x in gen_perp_mod])
    
    Delta1=min(norms)
    
    Delta1_p=min(np.linalg.norm(gen_perp,2, axis=1))
    
    reduced_basis = (1/np.sqrt(2))*np.array(simpleLLL(np.sqrt(2)*gen,0.25))
    
    gen_perp=np.linalg.inv(J@gen.T)
    
    gen_perp_mod=gen_perp-np.round(gen_perp@np.linalg.inv(gen))@gen
    gen_perp_mod[np.abs(gen_perp_mod)<1e-7]=0
    gen_perp_mod=gen_perp_mod[~np.all(gen_perp_mod == 0, axis=1)]*np.sqrt(2*np.pi)#/np.sqrt(np.pi)
    norms=np.asarray([np.linalg.norm(x,2) for x in gen_perp_mod])
    
    Delta2=min(norms)
    Delta2_p=min(np.linalg.norm(gen_perp,2, axis=1))
    
    return Delta1, Delta1_p, Delta2, Delta2_p 


def volume_unitcell_perp(L):

    
    J = Jsym(L**2)
    C = surfaceCode(L, signed=True) # instantiate auxiliary SC class
    #print(C.qubits, "\n")
    HX = C.HX() # X type parity check matrix
    HZ = C.HZ() # Z type parity check matrix
    MSurft = np.sqrt(np.pi)*np.block([[HX,np.zeros(np.shape(HX))],[np.zeros(np.shape(HX)),HZ]]) # corresponding generator matrix
    MSurf = (1./np.sqrt(2*np.pi)) * MSurft # rescaled generator matrix
    
    MGKPt = 2*np.sqrt(np.pi)*np.eye(2*L**2) # tilda generator matrix as in the notes --> rows are nullifiers
    MGKP = (1./np.sqrt(2*np.pi))*MGKPt # rescaled generator matrix --> the corresponding A matrix below is integer
    AGKP = MGKP@J@np.transpose(MGKP)
    
    Mtot = np.vstack([MGKP,MSurf])
    
    Lmat = np.sqrt(2)*Mtot # make the matrix integer to work with lattice functions

    Lmat = Matrix(Lmat.astype(int))
    
    U,H = hermite_normal_form(Lmat.T) # U is unimodular, the columns of H correspond to the new nullifiers


    # let's convert back to numpy float arrays
    U = np.array(U).astype(np.float64)
    H = np.array(H).astype(np.float64)

    newMmat = (1/np.sqrt(2))*H.T # this should be the new generator matrix, rows are the new independent generators
    gen = newMmat[~np.all(newMmat == 0, axis=1)]
    
    gen_perp=np.linalg.inv(J@gen.T)
    
    gen_perp_mod=gen_perp-np.round(gen_perp@np.linalg.inv(gen))@gen
    gen_perp_mod[np.abs(gen_perp_mod)<1e-7]=0
    gen_perp_mod=gen_perp_mod[~np.all(gen_perp_mod == 0, axis=1)]*np.sqrt(2*np.pi)
    norms=np.asarray([np.linalg.norm(x,2) for x in gen_perp_mod])
    Delta=min(norms)
    
    vol_N_sphere=vol_Nsphere(2*L**2,Delta/2 )
    
    
    return vol_N_sphere/np.linalg.det(gen_perp*np.sqrt(2*np.pi))


def hermite_reduce(M, scale):
    
    J = Jsym(int(M.shape[1]/2))
    
    Lmat = scale*M # make the matrix integer to work with lattice functions

    Lmat = Matrix(Lmat.astype(int))


    U,H = hermite_normal_form(Lmat.T) # U is unimodular, the columns of H correspond to the new nullifiers
    
    #print(U)
    #print(H)

    # let's convert back to numpy float arrays
    U = np.array(U).astype(np.float64)
    H = np.array(H).astype(np.float64)
    
    newMmat = (1/scale)*H.T # this should be the new generator matrix, rows are the new independent generators
    gen = newMmat[~np.all(newMmat == 0, axis=1)]

    gen_perp = np.linalg.inv(J @ gen.T)

    gen_perp_mod=gen_perp-np.round(gen_perp@np.linalg.inv(gen))@gen
    gen_perp_mod[np.abs(gen_perp_mod)<1e-7]=0
    
    gen_perp_mod=gen_perp_mod[~np.all(gen_perp_mod == 0, axis=1)]
    

    gen_perp_mod=gen_perp_mod*np.sqrt(2)
    
    norms=np.asarray([np.linalg.norm(x,2) for x in gen_perp_mod])
    
    Delta1=min(norms)
    
    return gen, gen_perp, Delta1

def check_voroni(x, M):
    M_length=np.diag(M@M.T)
    
    return np.all(np.abs(x@M.T)<=M_length/2)

#################################################################
# # Some useful functions for GKP lattices
#################################################################

def MGKP_n(n):
    return np.asarray([[np.sqrt(2*np.pi*n),0],[0,np.sqrt(2*np.pi*n)]])/np.sqrt(2*np.pi)

def logical_dim(M):
    #print('Jdim:',int(M.shape[1]/2))
    J = Jsym(int(M.shape[1]/2))
    A=M@J@M.T
    return np.sqrt(np.abs(np.linalg.det(A)))

def MGKP_hex(n):
    return np.sqrt(2/np.sqrt(3)*n)*np.asarray([[1,0],[1/2,np.sqrt(3)/2]])

#################################################################
# # Lattice completion
#################################################################


def surfLatticeGKPcompletion(Msurf,MGKP, check = False, verbose = False):
    '''
    Completes the surface code lattice generators to a GKP surface code.
    Starting from the lattice generated by surface code stabilizers, 
    checks if each GKP stabilizer is contained in the lattice and if not, adds it to the lattice generators. 
    
    input: # TODO add exception to handle errors from matrices that are not integer upon mult by \sqrt(2)
        Msurf: (N-1)x2N matrix, the lattice generator matrix for the rotated surface code on N modes
        MGKP: 2Nx2N matrix, the lattice generator matrix for a GKP qubit on each mode
        check: bool, it True checks linear dependence of the discarded GKP generators, default False 
        verbose: bool, if True prints additional messages, default False
    output:
        Mtot: 2Nx2N matrix, rows are a (possibly overcomplete) set of lattice generators for the GKP surface code
    '''
    
    Ltot = np.sqrt(2)*Msurf 
    LGKPleft = np.sqrt(2)*MGKP
    
                      
    Ltot = Ltot.astype(int)
    Ltot = Matrix(Ltot).T      
                      
    LGKPleft = LGKPleft.astype(int) # (transpose) integer matrix generating the lattice -> genetators are columns
    LGKPleft = Matrix(LGKPleft).T # (transpose) integer matrix of GKP lattice generators -> genetators are columns
                      
    while len(LGKPleft)>0: # if there are still GKP generators
        if verbose:
            print("{} GKP generators left".format(LGKPleft.shape[1]))
        sol = solve(Ltot,LGKPleft[:,0])
        

        if (sol != None): # check if the first lies on the lattice
            if verbose:
                print("found a GKP generator already in the lattice")
            if check:
                diff = Ltot@sol - LGKPleft[:,0]
                mismatch = diff.norm()
                if verbose:
                    print("solution verified, norm mismatch: ", mismatch)
                    print("the solution to the linear system is ", sol) # check that the solution is indeed integer
                if mismatch != 0:
                    print('linear dependence could not be verified, the norm mismatch is: {}; exiting'.format(mismatch))
                    return -1
            if verbose:
                print("found a dependent one")
            LGKPleft.col_del(0) # if yes, remove it from the list
        else:
            Ltot = Ltot.row_join(LGKPleft[:,0]) # otherwise add it to the generators
            LGKPleft.col_del(0) # then remove it from the list
    
    if verbose:
        print("\n reduction completed successfully, returning {} independent generators".format(Ltot.shape[1]))
    
    return (1/np.sqrt(2))*np.array(Ltot.T).astype(np.int64) # the returned matrix is an array of integers


#################################################################
# # Lattice completion: new, withour abelian dependency
#################################################################

def is_in_lattice(v,M):
    '''
    returns True if v is in the integer span of columns of M. If Mx == v then x is also returned, otherwise the second returned value is None
    '''
    sol = la.lstsq(M, v) # finds best approximation to v within the subspace of columns of M using least squares
    ans = False
    if np.allclose(M@sol, v) and np.allclose(sol, np.round(sol)): # if there is an integer solution x to Mx = v
        ans = True # then v is in the lattice
    return ans, sol
        
def CC_lattice_completion(Mqubit,Mbase, verbose = False): #### warning: needs to be completed!
    '''
    Completes the lattice generator of a concatenated code given qubit code and mode-wise GKP lattice generator.
    Starting from the lattice generated by qubit code stabilizers, 
    checks if each GKP stabilizer is contained in the lattice and if not, adds it to the lattice generators. 
    
    input:
        Mqubit: (N-k)x2N matrix, lattice generator for qubit code with k logical qubits over N modes
        Mbase: 2x2 matrix, the lattice generator matrix for a GKP qubit on each mode
        verbose: bool, if True prints additional messages, default False
    output:
        Mtot: 2Nx2N matrix, rows are a (possibly overcomplete) set of lattice generators for the GKP surface code
        
        
    CONVENTIONS: 
        each row of the input qubit generator corresponds to Pauli strings: entries are in {0,1,-1}, first N entries correspond to X, last N entries to Z. Entries are exponent of the operators. Example: for 3 qubits with first stabilizer XIZ^\dagger we have M_1 = (1,0,0,0,0,-1).
    '''
    
    Ltot = Mqubit # rows correspond to stabilizers
    LGKPleft = Mbase # builds generator of base code for all modes given single mode generator
    
                      
    while len(LGKPleft)>0: # if there are still GKP generators
        if verbose:
            print("{} GKP generators left".format(LGKPleft.shape[0]))
            
        member, sol = is_in_lattice(LGKPleft[0],Ltot.T) # check if the vector is an integer lin comb of the generators already found
        if member: # if the first lies on the lattice
            if verbose:
                print("found a GKP generator already in the lattice")
            if check:
                diff = sol@Ltot - LGKPleft[0]
                mismatch = diff.norm()
                if verbose:
                    print("solution verified, norm mismatch: ", mismatch)
                    print("the solution to the linear system is ", sol) # check that the solution is indeed integer
                if np.abs(mismatch) > 10**(-8):
                    print('linear dependence could not be verified, the norm mismatch is: {}; exiting'.format(mismatch))
                    return -1
        else:
            Ltot = Ltot.vstack(LGKPleft[0]) # otherwise add it to the generators
            
        LGKPleft.delete(LGKPleft,0,0) # remove the current GKP stabilizer from the list of those to check
        
    if verbose:
        print("\n reduction completed successfully, returning {} independent generators".format(Ltot.shape[1]))
    
    return Ltot 



#########################
###Lattice algorithms and LLL v2
#########################
def pi_proj(x, i, Bs):
    """
    projection on GS basis B_s
    """
    # print("BsBsT",Bs@Bs.T)
    # print("norm",np.linalg.norm(Bs,2,axis=1)**2)
    normed_Bs = Bs / (np.linalg.norm(Bs, 2, axis=1) ** 2)[:, None]
    # print("norm",normed_Bs)

    n = Bs.shape[1]

    components = np.concatenate([np.zeros((i, n)), normed_Bs[i:]], axis=0) @ x

    return components @ Bs


def GS_J(B):
    """
    Gram schmidt-orthogonalization
    """
    n = len(B)

    GS_B = np.zeros_like(B)
    for i in range(n):
        GS_B[i] = B[i] - (GS_B[:i] @ B[i] / np.diag(GS_B[:i] @ GS_B[:i].T)) @ GS_B[:i]

    return GS_B


def Nearest_Plane(B, t):
    """
    Babais Nearest Plano Algo as in the Crypto Course
    """
    if len(B) == 0:
        return np.zeros(len(t))

    else:
        Bs = GS_J(B)
        # print("Bs", Bs)
        c = np.round(t @ Bs[-1] / np.linalg.norm(Bs[-1]) ** 2)

    return c * B[-1] + Nearest_Plane(np.delete(B, -1, 0), t - c * B[-1])


def SizeReduce(B):
    """
    see Crypto Course
    """
    B_temp = B.copy()  # make hardcopy to not modify input matrix

    Bs = GS_J(B_temp)
    for i in range(1, len(B_temp)):
        x = Nearest_Plane(B_temp, B_temp[i] - Bs[i])
        # print("x",x)
        B_temp[i] -= x
    return B_temp


def LLL_Crypt(B, delta):
    """
    LLL algorithm see Crypto Course
    """
    B_SR = SizeReduce(B)
    # print("checking",B_SR)
    Bs = GS_J(B)

    for i in range(len(B_SR) - 1):
        if delta * np.linalg.norm(pi_proj(B_SR[i], i, Bs)) ** 2 > np.linalg.norm(pi_proj(B_SR[i + 1], i, Bs)) ** 2:
            B_SR[[i, i + 1]] = B_SR[[i + 1, i]]

            return LLL_Crypt(B_SR, delta)

    return B_SR

def RoundingCVP(B, Binv,t):
    """
    naive rounding trick to solve CVP on row-vector by Babai.
    """
    return np.round(t@Binv)@B


def Round_Sublattice(t):
    """
    Rounding tric on bare-GKP orthogonal sublattice.
    """
    return np.round(t)

#####
##test for Lattice algorithms and LLL v2
#####

def test_proj_and_GS(A):
    As = GS_J(A)
    for i in range(len(A)):
        assert np.allclose(As[i], pi_proj(A[i], i, As)), "something wrong with GS/proj at %i" % (i)

    return 0


def test_LLL(A, delta):
    LLL_A = LLL_Crypt(A, delta)

    transition = A @ np.linalg.inv(LLL_A)

    assert np.allclose(transition, np.round(transition)), "transition is not integer"
    assert np.isclose(np.abs(np.linalg.det(transition)), 1), "transition is not unimodular"

    return 0


def test_GS(A):
    As = GS_J(A)

    As_AsT = As @ As.T

    assert np.allclose(As_AsT, np.diag(np.diag(As_AsT))), "GS does not return orthogonal matrix"

    return 0



########################
#classical code algorithms
########################


def gen_standard_form(G):
    G = G.astype(int)
    lr, lc = G.shape

    j0 = 0
    ##determine size of upper left identity matrix
    for x in range(lr):
        if np.all(G[:x, :x] == np.eye(x, dtype=int)):
            j0 = x

    # reduce lower left block to 0
    for x_2 in range(j0, lr):
        if np.any(G[x_2, :j0] != 0):
            indices = np.where(G[x_2, :j0] == 1)[0]
            # print(G)
            # print(G[x_2])
            # print(indices)
            for ind in indices:
                G[x_2] += G[ind]
    G = G % 2

    # print("start j=",j0)

    # print(G)

    Cswaps = []

    for j in range(j0, lr):
        # print('j=',j)
        # step 1
        G = G.astype(int)
        if G[j, j] == 0:
            flag = 0  ###flag !=0 if G_jj=0 and G_ij=0 for all i>j
            for x_3 in range(j + 1, lr):
                # print("x_3",x_3)
                if G[x_3, j] != 0:
                    G[[x_3, j]] = G[[j, x_3]]
                    flag += 1
                    # print("flagging")
                    break

            # print("flagged", flag)
            if flag == 0:
                for h in range(lc):
                    # print("gjh", G[j,h])
                    if G[j, h] != 0:
                        G[:, [j, h]] = G[:, [h, j]]
                        Cswaps += [[j, h]]
                        # print("column swap",j,h)
                        break
        # step 2
        G = G.astype(float)
        G[j] /= G[j, j]

        # print("step 2 complete")
        assert (G[j, j] == 1)
        # print((G%2).astype(int))

        # step 3
        for x_4 in range(lr):
            if x_4 != j:
                G[x_4] = G[x_4] - G[x_4, j] * G[j]

    # print("final G")
    # print((G%2).astype(int))

    return (G % 2).astype(int), Cswaps


def H_from_G(G):
    k, n = G.shape
    G_standard, Cswaps = gen_standard_form(G)
    # print("standard G")
    # print(G_standard)
    A = G_standard[:, k:]
    # print("A")
    # print(A)
    # print(A.shape)
    # print(k, n-k)

    H = np.concatenate((-A.T, np.eye(n - k)), axis=1)

    for c in Cswaps[::-1]:
        H[:, [c[0], c[1]]] = H[:, [c[1], c[0]]]
    # print("H")
    # print(H)
    return (H % 2).astype(int)


def is_in_G(c, H):
    return np.all((c @ H.T % 2) == 0)



###################################
##Construction A C decomposition
###################################

def decompose_L1(G):
    H_G = H_from_G(G)

    for j in range(G.shape[1]):
        G_prime = np.delete(G, j, axis=1)

        G0 = np.insert(G_prime, j, np.zeros(len(G)), axis=1)
        G1 = np.insert(G_prime, j, np.ones(len(G)), axis=1)

        check1 = np.all(G0 @ H_G.T % 2 == 0)
        check2 = np.all(G1 @ H_G.T % 2 == 0)

        if check1 and check2:
            return j
    print("FAIL")
    return None


def decompose_L2(G):
    check_par = np.ones(2)
    H_G = H_from_G(G)

    for j1 in range(G.shape[1]):
        for j2 in range(G.shape[1]):
            if j1 > j2:

                G_p = np.delete(G, [j1, j2], axis=1)
                G_pp = G[:, [j1, j2]]

                if np.all(G_pp @ check_par % 2 == 0):

                    G_00 = G.copy()
                    G_00[:, j1] = np.zeros(len(G))
                    G_00[:, j2] = np.zeros(len(G))

                    G_11 = G.copy()
                    G_11[:, j1] = np.ones(len(G))
                    G_11[:, j2] = np.ones(len(G))

                    check1 = np.all(G_00 @ H_G.T % 2 == 0)
                    check2 = np.all(G_11 @ H_G.T % 2 == 0)

                    if check1 and check2:
                        return j1, j2
    print("FAIL")
    return None

##########Theta function sampling#############

from numpy.random import normal
from numpy.random import randint
from numpy.random import choice
from numpy.random import rand

def gaussian(x, s, c):
    return np.exp(-(x-c)**2/(2*s**2))#/np.sqrt(2*np.pi*s**2)

def sampleZ(s, c, t):
    """
    s=standard deviation
    c=mean
    t=function of O(sqrt(log(2n)))
    """

    while True:
        #print("boundaries!", c-s*t, c+s*t)
        sample=randint(np.ceil(c-s*t),np.floor(c+s*t))
        prob=gaussian(sample,s, c)
        random_var=rand()
        if random_var<=prob:
            return sample



def sampleD(s, c, t, B):
    GSB=GS_J(B)

    normalization=np.diag(GSB@GSB.T)
    c_prime=GSB@c/normalization
    s_prime=s/np.sqrt(normalization)
    c_=c[:]
    v_=0*c_

    for i in range(len(B)-1,0,-1):
        z_i=sampleZ(s_prime, c_prime[i], t)
        c_=c_-z_i*B[i]
        v_=v_+z_i*B[i]

    return v_


class sampleD(object):
    def __init__(self, sigma, mean, t, B, GSB):
        self.mean = mean
        self.sigma = sigma
        self.t = t
        self.B = B
        self.len = len(B)
        self.GSB = GSB
        # print(self.GSB)

        self.normalization = np.diag(self.GSB @ self.GSB.T)
        # self.c_prime=self.GSB@self.mean/self.normalization

        self.s_prime = self.sigma / np.sqrt(self.normalization)
        self.c_ = self.mean[:]
        self.v_ = 0 * self.c_

    def samp(self):
        c_ = self.c_.copy()
        v_ = self.v_.copy()

        for i in reversed(range(1, self.len)):
            c_prime_i = self.c_ @ self.GSB[i].T / self.normalization[i]
            # print(c_prime_i)

            z_i = sampleZ(self.s_prime[i], c_prime_i, self.t)

            c_ = c_ - z_i * self.B[i]
            v_ = v_ + z_i * self.B[i]

        return v_


class Theta_decode(object):
    """
    Evaluates Theta function of shifted lattice belonging to each logical shift+initial correction.
    Uses Poisson Formula such that the MLD probability is estimated as average of phase over discrete lattice gaussian, as in


    """

    def __init__(self, sigma, t, B, logicals, prestore_samples):
        self.sigma = sigma
        self.t = t
        self.B = B  ####dual basis
        self.len = len(B)
        self.GSB = GS_J(B)
        self.sampler = sampleD(1/np.sqrt(2*np.pi*self.sigma**2), np.zeros(len(B)), t, B, self.GSB)
        self.N = int(len(B)/2)  # number of modes
        self.logicals = np.append(np.asarray(logicals), np.zeros((1,2*self.N)), axis=0)  ###logical shifts
        self.J = Jsym(np.int(self.N))  # symplectic form
        self.pre_samples = np.asarray([self.sampler.samp() for s in range(prestore_samples)])
        self.partial_sum = np.zeros(len(self.logicals))

    def decode(self, num_samples, init_correction):
        shifts = init_correction + self.logicals  ####size(#logicals, 2n)
        samples = np.asarray([self.sampler.samp() for s in range(num_samples)])  ##size=(num_samples, 2n)

        prod = -1j * 2 * np.pi * shifts @ self.J @ samples.T  ##shifts=(#logicals, 2n), samples.T=(2n, num_samples)
        sample_classes = np.sum(np.cos(prod), axis=-1)

        return self.logicals[np.where(sample_classes == np.max(sample_classes))[0][0]]

    def decode_prestore(self, init_correction):
        shifts = init_correction + self.logicals  ####size(#logicals, 2n)
        samples = self.pre_samples  ##size=(prestore_samples, 2n)

        prod = -1j * 2 * np.pi * shifts @ self.J @ samples.T  ##shifts=(#logicals, 2n), samples.T=(2n, prestore_samples)
        sample_classes = np.sum(np.cos(prod), axis=-1)

        return self.logicals[np.where(sample_classes == np.max(sample_classes))[0][0]]

    def add_sample(self, init_correction):
        shifts = init_correction + self.logicals  ####size(#logicals, 2n)
        sample = self.sampler.samp()
        prod = -2 * np.pi * shifts @ self.J @ sample  ##shifts=(#logicals, 2n), samples.T=(2n, num_samples)
        sample_classes = np.real(np.cos(prod))
        # print("shifts", shifts)
        # print("sample", sample)
        # print("prod", prod)
        # print("SC", sample_classes)
        # print(sample_classes.shape)

        self.partial_sum += sample_classes
        # print("PS",self.partial_sum)

        return 0


class symplectic_dec:
    "decoder based on symplectic equivalence i our paper. Transform code+error to square GKP code, solve CVP there and transform back"
    def __init__(self, M):
        self.J = Jsym(int(len(M) / 2))
        self.M_square = np.diag(np.ones(len(M)))
        self.M_square[0, 0] *= np.sqrt(2)
        self.M_square[1, 1] *= np.sqrt(2)
        self.ST = np.linalg.inv(M) @ self.M_square
        self.S = self.ST.T
        self.Sinv = np.linalg.inv(self.S)

        self.M_square_dual_inv = self.J @ self.M_square.T
        self.M_square_dual = np.linalg.inv(self.M_square_dual_inv)

        return

    def decode(self, t):
        CVPsquare = RoundingCVP(self.M_square_dual, self.M_square_dual_inv, self.Sinv @ t)

        return self.S @ CVPsquare























