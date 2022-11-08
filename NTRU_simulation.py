#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
from numpy.random import seed
from gridcodefunctions import *
import multiprocessing
from multiprocessing import Pool
import itertools
from time import time

def balancedmod_ar(ar,q):
    return (ar+q/2)%q-q/2


def rbalancedmod_ar(ar,q):
    return (ar+q//2)%q-q//2



def T(f):
    temp=np.zeros((len(f), len(f)))
    for i in range(len(f)):
        temp[i]=np.roll(f,i)
        
    return temp

def That(f):
    temp=np.zeros((len(f), len(f)))
    for i in range(len(f)):
        temp[i]=np.roll(f,-i)
        
    return temp

def sigma(f):
    temp=np.zeros(len(f))
    temp[0]=f[0]
    temp[1:]=f[1:][::-1]
    
    return temp

def Tsigma(f):
    temp=np.zeros((len(f), len(f)))
    for i in range(len(f)):
        temp[i]=sigma(np.roll(f,i))
    
    return temp

def Thatsigma(f):
    temp=np.zeros((len(f), len(f)))
    for i in range(len(f)):
        temp[i]=sigma(np.roll(f,-i))
    
    return temp



def pub_basis(h,q,n, l):
    pk=That(h)
    print("make pub")
    print(n,pk.shape)
    return np.sqrt(l/q)*np.block([[np.eye(n),pk],[np.zeros((n,n)), q*np.eye(n)]])

def sec_basis(f,g,q,n,l):
    f_sigma=Tsigma(f)
    Tg=T(g)
    return np.sqrt(l/q)*np.block([[f_sigma,Tg]])#,[np.zeros((n,n)), q*np.eye(n)]])

def Xsec_basis(f,g,q,n,l):
    f_sigma=Tsigma(f)
    Tg=T(g)
    return np.sqrt(l/q)*np.block([[Tg, f_sigma]])

def round_CVP(sec_bas, sec_inv, vec):
    
    CVP=np.round(vec@sec_inv)@sec_bas
    
    return vec-CVP

def Jsym(n):
    J1=np.asarray([[0,1],[-1,0]])
    J=np.kron(J1,np.eye(n))
    return J


from numpy.random import normal
def sample_err(n,sig,bias,shots):
    dq=normal(0,sig*bias,size=(shots,n))
    dp=normal(0,sig/bias,size=(shots,n))
    
    return np.concatenate([dq,dp],axis=1)

def syndrome(e,M,n):
    return np.mod(M@Jsym(n)@e,1)

def triv_syndrome(e, scale):
    return balancedmod_ar(e,1)


# In[96]:


def NTRU_dec(c,Tf,Thinv,q,p_list):
    
    dims=Tf.shape[0]
    moduli=np.asarray(p_list)
    N=np.prod(moduli)
    reduced_a=np.zeros((len(p_list),dims))
    a=rbalancedmod_ar(Tf@c,q)
    
    for p_ind in range(len(moduli)):

        reduced_a[p_ind]=a%moduli[p_ind]
        
    CRT=np.zeros(dims)
    
    

    for i in range(dims):
        #print("moduli")
        #print(moduli)
        #print("a_red")
        #print(reduced_a)
        CRT[i]=chinese_remainder(moduli, reduced_a[:,i])
    #print("CRT")
    #print(CRT)
    
    m_bal=rbalancedmod_ar(CRT,N)
    #print("mbal")
    #print(m_bal)
    
    b=rbalancedmod_ar((c-m_bal),q)
    b=Thinv@b
    r=rbalancedmod_ar(b,q)
    #r=sigma(r)
        
        
    out=np.concatenate([-r, m_bal])
    #out=rbalancedmod_ar(out,N)
    
    #print("NTRUdec")
    #print(out)
    return out


from functools import reduce
def chinese_remainder(m, a):
    
    sum = 0
    prod = reduce(lambda acc, b: acc*b, m)
    for n_i, a_i in zip(m, a):
        p = prod // n_i
        sum += a_i * mul_inv(p, n_i) * p
    return sum % prod
 
 
 
def mul_inv(a, b):
    b0 = b
    x0, x1 = 0, 1
    if b == 1: return 1
    while a > 1:
        q = a // b
        a, b = b, a%b
        x0, x1 = x1 - q * x0, x0
    if x1 < 0: x1 += b0
    return x1




def simulateNTRU(sig,M,Tsigma_f,Tsigma_hinv,s_basis,MJ_inv, n, q,p, l,NUM_err, NUM_max, Babai, precision=8):

    NUM_tot=0
    count_err=0
    count_CHECK=0
    count_Babai=0
    J=Jsym(n)
    #print(sig)
    seed(int(sig * 200*Tsigma_f[2,2] + int(time())) % 127239)
    while count_err<NUM_err and NUM_tot<NUM_max:
        #print("N:",NUM_tot)
        NUM_tot+=1

        err=sample_err(n,sig,1,1)[0]
        #print("err")
        #print(err)
        
        triv_synd=triv_syndrome(np.sqrt(2*q)*err, 1)

        err_prime=err-triv_synd/np.sqrt(2*q)
        #print("err_prime", err_prime)

        synd=balancedmod_ar(M@J@err_prime ,1)
        qsynd=np.round(q*synd)
        #print("qsynd")
        #print(qsynd)

        dec_res=NTRU_dec(qsynd[:n],Tsigma_f,Tsigma_hinv,q,p)
        #print("dec_res", dec_res)

        err_pp=err_prime-dec_res/np.sqrt(2*q)

        #print("decres", dec_res/np.sqrt(2*q))
        #print("err_pp", err_pp)
        prod=np.round(M@J@err_pp/2,precision)

        syndpp=(2*prod)%1
        logpp=prod%1

       	CHECK=np.allclose(syndpp%1, 0,atol=1/10**precision)
        #print("synd")
        #print(syndpp)
        #print("CHECK", CHECK)

        if CHECK==False and Babai==True:
            
            count_Babai+=1
            print("Babai", NUM_tot)
            synd=synd%1
            corr_bab0=MJ_inv@synd
            corr1=Nearest_Plane(s_basis/2*np.sqrt(2*q), corr_bab0*np.sqrt(2*q))/np.sqrt(2*q)
            corr=corr_bab0-corr1
            
            err_pp=err_prime-corr
            prod = np.round(M @ J @ err_pp / 2, precision)
            syndpp = 2*prod%1
            logpp = prod % 1

            CHECK = np.allclose(syndpp % 1, 0, atol=1/10**precision)

            #print("synd after babai")
            #print(syndpp)



        if not CHECK:
            print("FAIL")
            print(syndpp)
            
            
        
        
        
        flag=np.any(np.isclose(logpp%1,0.5))#1-np.allclose(logpp%1, 0)

        count_err+=flag*CHECK
        count_CHECK+=1-CHECK


    prob_check=count_CHECK/NUM_tot
    prob_err=count_err/NUM_tot
    rat_Babai=count_Babai/NUM_tot
   
    #print("----------------------------------------------------",sig,":",prob_err, prob_check, rat_Babai,"----------------------------------------------------")
 
    return prob_err, prob_check, rat_Babai


def simulateBabai(sig, M, Tsigma_f, Tsigma_hinv, s_basis, MJ_inv, n, q, p, l, NUM_err, NUM_max, Babai=True, precision=8):
    NUM_tot = 0
    count_err = 0
    count_CHECK = 0
    count_Babai = 0
    J = Jsym(n)
    # print(sig)
    seed(int(sig * 200 * Tsigma_f[2, 2] + int(time())) % 127239)
    while count_err < NUM_err and NUM_tot < NUM_max:

        # print("N:",NUM_tot)
        NUM_tot += 1
        err = sample_err(n, sig, 1, 1)[0]
        # print(err)

        synd = balancedmod_ar(M @ J @ err, 1)

        corr_bab0 = MJ_inv @ synd
        corr1 = Nearest_Plane(s_basis / 2 , corr_bab0)
        corr = corr_bab0 - corr1

        err_pp = err_prime - corr
        prod = np.round(M @ J @ err_pp / 2, precision)
        syndpp = 2 * prod % 1
        logpp = prod % 1

        CHECK = np.allclose(syndpp % 1, 0, atol=1/10**precision)

            # print("synd after babai")
            # print(syndpp)

        if not CHECK:
            print("FAIL")
            print(syndpp)

        flag = np.any(np.isclose(logpp % 1, 0.5))

        count_err += flag * CHECK
        count_CHECK += 1 - CHECK

    prob_check = count_CHECK / NUM_tot
    prob_err = count_err / NUM_tot

    # print(sig,":",prob_err, prob_check, rat_Babai)

    return prob_err, prob_check, 1


def simulateNTRU_bab2(sig, M, Tsigma_f, Xs_basis, n, q, l, NUM_err, NUM_max,precision=8):
    NUM_tot = 0
    count_err = 0
    count_CHECK = 0
    J = Jsym(n)

    while count_err < NUM_err and NUM_tot < NUM_max:
        # print(NUM_tot)
        NUM_tot += 1
        err = sample_err(n, sig, 1, 1)[0]

        triv_synd = triv_syndrome(np.sqrt(2 * q) * err, 1)

        err_prime = err - triv_synd / np.sqrt(2 * q)

        synd = balancedmod_ar(M @ J @ err_prime, 1)
        qsynd = np.round(q * synd) % q

        dec_res = Nearest_Plane(Xs_basis, qsynd)

        err_pp = err_prime - dec_res / np.sqrt(2 * q)

        logpp = np.round(np.mod(M @ J @ err_pp / 2, 1), precision)

        syndpp = (2 * logpp) % 1

        CHECK = np.allclose(syndpp % 1, 0, atol=1/10**precision)
        # print("synd", syndpp)

        # print("CHECK",CHECK)

        # print(logpp)

        flag = np.any(np.isclose(logpp, 0.5))
        # print(flag)

        count_err += flag * CHECK
        count_CHECK += 1 - CHECK

    prob_check = count_CHECK / NUM_tot
    prob_err = count_err / NUM_tot
    # rat_Babai=count_Babai/NUM_tot

    return prob_err, prob_check, 1


from tqdm import tqdm

class Cload_and_sim(object):

    def __init__(self, sigmas, n,q,p_list,d,prefix, NUM_ERR, NUM_MAX, Babai=True, precision=8):

        self.NUM_ERR=NUM_ERR
        self.NUM_MAX=NUM_MAX

        self.Babai=Babai
        self.precision=precision
        self.sigmas=sigmas
        self.n=n
        self.p_list=p_list
        self.d=d
        self.prefix=prefix
        self.l = 2
        self.p = np.prod(np.asarray(self.p_list))
        self.q=q
        self.data = np.genfromtxt("NTRU_%s_n%i_q%i_p%i_d%i.txt" % (self.prefix, self.n, self.q, self.p, self.d))
        self.h, self.h_inv, self.f_vec, self.g_vec = self.data
        self.T_bar_h = That(self.h)

        self.Tsigma_hinv = Tsigma(self.h_inv)
        self.Tsigma_f = Tsigma(self.f_vec)

        self.M = pub_basis(self.h, self.q, self.n, 2)

        self.MJ_inv = np.linalg.inv(self.M @ Jsym(n))
        self.s_basis = sec_basis(self.f_vec, self.g_vec, self.q, self.n, self.l)
        self.Xs_basis = Xsec_basis(self.f_vec, self.g_vec, self.q, self.n, self.l)

        self.p_err = np.zeros(len(sigmas))
        self.p_check = np.zeros(len(sigmas))
        self.rat_Babai = np.zeros(len(sigmas))

    def evaluate(self, i):
        data = simulateNTRU(self.sigmas[i], self.M, self.Tsigma_f, self.Tsigma_hinv, self.s_basis, self.MJ_inv, self.n,
                            self.q, self.p_list, self.l, self.NUM_ERR, self.NUM_MAX, Babai=self.Babai, precision=self.precision)

        return data

    def Babai_evaluate(self, i):
        data = simulateBabai(self.sigmas[i], self.M, self.Tsigma_f, self.Tsigma_hinv, self.s_basis, self.MJ_inv,
                            self.n,
                            self.q, self.p_list, self.l, self.NUM_ERR, self.NUM_MAX, Babai=self.Babai, precision=self.precision)

        return data


    def Babai2_evaluate(self, i):
        data = simulateNTRU_bab2(self.sigmas[i], self.M, self.Tsigma_f, self.Xs_basis, self.n, self.q, self.l, self.NUM_ERR, self.NUM_MAX, precision=self.precision)

        return data


    def save(self, path):
        tosave = np.asarray([self.sigmas, self.p_err, self.p_check, self.rat_Babai])
        np.savetxt("%s/NTRU_sim_NTRUDEC_B%i_%s_n%i_q%i_p%i_d%i_NERR%i.txt" % (
        path, self.Babai, self.prefix, self.n, self.q, self.p, self.d, self.NUM_ERR), tosave,
                   comments="#NUM_ERR=%i, NUM_MAX=%i, Babai=%i" % (self.NUM_ERR, self.NUM_MAX, self.Babai))

        plt.figure("checking n%i, q%i,p%i" % (self.n, self.q, self.p))
        plt.plot(self.sigmas, self.p_err, label="p_err")
        plt.plot(self.sigmas, self.p_check, label="p_check")
        plt.legend()
        plt.savefig("%s/checking_n%i_q%i_p%i.pdf" % (path, self.n, self.q, self.p))

        print(self.n, self.d, self.q, "DONE")

        return 0

# In[160]:

cpu_count=multiprocessing.cpu_count()

Ps=[[3],[5],[7],[3,5]]
Qs=[4,8,16,32]
Ns=[7,11,17,53]
Ds=[5,7,11,29]
NUM_ERR=int(100)
NUM_MAX=int(1e5)


sigmas=np.linspace(1e-7,0.2,100)

if __name__ == '__main__':

    for k in range(len(Ns)):
        n,q,d,p_list=Ns[k], Qs[k], Ds[k], Ps[k]
        worker=Cload_and_sim(sigmas, n,q,p_list,d,"opt", NUM_ERR, NUM_MAX, Babai=True, precision=8)

        with Pool(cpu_count) as p:
            data=p.map(worker.evaluate, list(range(len(sigmas))))

        for i in range(len(sigmas)):
            worker.p_err[i], worker.p_check[i], worker.rat_Babai[i] = data[i][0], data[i][1], data[i][2]

        worker.save("results_new_y")





