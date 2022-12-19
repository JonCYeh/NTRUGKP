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



def pub_basis(h,q,n):
    pk=That(h)
    print("make pub")
    print(n,pk.shape)
    return np.block([[np.eye(n),pk],[np.zeros((n,n)), q*np.eye(n)]])

def sec_basis(f,g,q,n):
    f_sigma=Tsigma(f)
    Tg=T(g)
    return np.block([[f_sigma,Tg]])#,[np.zeros((n,n)), q*np.eye(n)]])

def Xsec_basis(f,g,q,n,l):
    f_sigma=Tsigma(f)
    Tg=T(g)
    return np.block([[Tg, f_sigma]])

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


def NTRU_dec(c, That_sigma_f, T_sigma_hinv, q, p_list):
    dims = That_sigma_f.shape[0]
    moduli = np.asarray(p_list)
    N = np.prod(moduli)
    reduced_a = np.zeros((len(p_list), dims))
    a = rbalancedmod_ar(That_sigma_f @ c, q)

    # print(That_sigma_f%3)

    for p_ind in range(len(moduli)):
        reduced_a[p_ind] = a % moduli[p_ind]  ##  v % p_i

    CRT = np.zeros(dims)

    for i in range(dims):
        # print("moduli")
        # print(moduli)
        # print("a_red")
        # print(reduced_a)
        CRT[i] = chinese_remainder(moduli, reduced_a[:, i])
    # print("CRT")
    # print(CRT)

    v = sigma(rbalancedmod_ar(CRT, N))
    # print("mbal")
    # print(m_bal)

    b = rbalancedmod_ar((v - c), q)
    r = T_sigma_hinv @ b
    r = rbalancedmod_ar(r, q)
    u = sigma(r)

    out = np.concatenate([u, v])
    # out=rbalancedmod_ar(out,N)

    # print("NTRUdec")
    # print(out)
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


def NTRU_dec(c, That_sigma_f, That_sigma_hinv, q, p_list):
    dims = That_sigma_f.shape[0]
    moduli = np.asarray(p_list)
    N = np.prod(moduli)
    reduced_a = np.zeros((len(p_list), dims))
    a = rbalancedmod_ar(That_sigma_f @ c, q)

    # print(That_sigma_f%3)

    for p_ind in range(len(moduli)):
        reduced_a[p_ind] = a % moduli[p_ind]  ##  v % p_i

    CRT = np.zeros(dims)

    for i in range(dims):
        # print("moduli")
        # print(moduli)
        # print("a_red")
        # print(reduced_a)
        CRT[i] = chinese_remainder(moduli, reduced_a[:, i])
    # print("CRT")
    # print(CRT)

    v = rbalancedmod_ar(CRT, N)
    # print("mbal")
    # print(m_bal)

    b = rbalancedmod_ar((v - c), q)
    r = That_sigma_hinv @ b
    r = rbalancedmod_ar(r, q)
    u = sigma(r)

    out = np.concatenate([u, v])
    # out=rbalancedmod_ar(out,N)

    # print("NTRUdec")
    # print(out)
    return out


from functools import reduce


def chinese_remainder(m, a):
    sum = 0
    prod = reduce(lambda acc, b: acc * b, m)
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
        a, b = b, a % b
        x0, x1 = x1 - q * x0, x0
    if x1 < 0: x1 += b0
    return x1


def simulateNTRU(sig, M, That_sigma_f, That_sigma_hinv, n, q, p, NUM_max, precision=12):
    NUM_tot = 0
    count_err = 0
    count_CHECK = 0
    J = Jsym(n)
    # print(sig)
    seed(int(sig * 200 * That_sigma_f[2, 2] + int(time())) % 127239)
    while NUM_tot < NUM_max:

        NUM_tot += 1

        err = sample_err(n, sig, 1, 1)[0]
        # print("err")
        # print(err)

        triv_synd = triv_syndrome(np.sqrt(2 * q) * err, 1)

        err_prime = err - triv_synd / np.sqrt(2 * q)
        # print("err_prime", err_prime)

        synd = balancedmod_ar(M @ J @ err_prime, 1)
        qsynd = np.round(q * synd, 12)
        # print("qsynd")
        # print(qsynd)

        dec_res = NTRU_dec(qsynd[:n], That_sigma_f, That_sigma_hinv, q, p)
        # print("dec_res", dec_res)

        err_pp = err_prime - dec_res / np.sqrt(2 * q)

        # print("decres", dec_res/np.sqrt(2*q))
        # print("err_pp", err_pp)
        prod = np.round(M @ J @ err_pp / 2, precision)

        syndpp = (2 * prod) % 1
        logpp = prod % 1

        CHECK = np.allclose(syndpp, 0, atol=10 ** (-1 * precision))

        if not CHECK:
            print("FAIL")
            print(syndpp)

        flag = np.any(np.isclose(logpp % 1, 0.5, atol=10 ** (-1 * precision)))

        count_err += flag * CHECK
        count_CHECK += 1 - CHECK

    prob_check = count_CHECK / NUM_tot
    prob_err = count_err / NUM_tot

    return prob_err, prob_check


def simulateNTRU_babX(sig, M, Xs_basis, n, q, NUM_max, precision=12):
    NUM_tot = 0
    count_err = 0
    count_CHECK = 0
    J = Jsym(n)

    while NUM_tot < NUM_max:
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

        CHECK = np.allclose(syndpp % 1, 0, atol=10 ** (-1 * precision))
        # print("synd", syndpp)

        # print("CHECK",CHECK)

        # print(logpp)

        flag = np.any(np.isclose(logpp % 1, 0.5, atol=10 ** (-1 * precision)))
        # print(flag)

        count_err += flag * CHECK
        count_CHECK += 1 - CHECK

    prob_check = count_CHECK / NUM_tot
    prob_err = count_err / NUM_tot

    return prob_err, prob_check


from tqdm import tqdm


class Cload_and_sim(object):

    def __init__(self, sigmas, n, q, p_list, d, prefix, folder, NUM_MAX, precision=8, inv=False):

        self.folder = folder
        self.inv = inv
        self.NUM_MAX = NUM_MAX

        self.precision = precision
        self.sigmas = sigmas
        self.n = n
        self.p_list = p_list
        self.d = d
        self.prefix = prefix
        self.l = 2
        self.p = np.prod(np.asarray(self.p_list))
        self.q = q
        self.data = np.genfromtxt(
            "%s/NTRU_%s_n%i_q%i_p%i_d%i.txt" % (self.folder, self.prefix, self.n, self.q, self.p, self.d))

        if inv:
            self.h, self.h_inv, self.f_vec, self.g_vec, _ = self.data
            self.That_sigma_hinv = Thatsigma(self.h_inv)
        else:
            self.h, self.f_vec, self.g_vec, _ = self.data

        self.T_bar_h = That(self.h)

        self.That_sigma_f = Thatsigma(self.f_vec)

        self.M = np.sqrt(self.l / self.q) * pub_basis(self.h, self.q, self.n)

        self.MJ_inv = np.linalg.inv(self.M @ Jsym(n))
        self.Xs_basis = Xsec_basis(self.f_vec, self.g_vec, self.q, self.n, self.l)
        self.Xs_HKZ_basis = np.genfromtxt(
            "%s/NTRU_HKZ_n%i_q%i_p%i_d%i.txt" % (self.folder, self.n, self.q, self.p, self.d))

        self.p_err = np.zeros(len(sigmas))
        self.p_check = np.zeros(len(sigmas))
        self.rat_Babai = np.zeros(len(sigmas))

    def evaluate(self, i):
        assert (self.inv == True)
        data = simulateNTRU(self.sigmas[i], self.M, self.That_sigma_f, self.That_sigma_hinv, self.n,
                            self.q, self.p_list, self.NUM_MAX, precision=self.precision)
        return data

    def BabaiX_evaluate(self, i):
        data = simulateNTRU_babX(self.sigmas[i], self.M, self.Xs_HKZ_basis, self.n, self.q,
                                 self.NUM_MAX, precision=self.precision)
        return data

    def save(self, path, label):
        tosave = np.asarray([self.sigmas, self.p_err, self.p_check])
        np.savetxt("%s/NTRU_sim_%s_%s_n%i_q%i_p%i_d%i_NMAX%i.txt" % (
            path, label, self.prefix, self.n, self.q, self.p, self.d, self.NUM_MAX), tosave)

        plt.figure("checking_%s_n%i_q%i_p%i" % (label, self.n, self.q, self.p))
        plt.plot(self.sigmas, self.p_err, label="p_err")
        plt.plot(self.sigmas, self.p_check, label="p_check")
        plt.legend()
        plt.savefig("%s/%s_n%i_q%i_p%i.pdf" % (path, label, self.n, self.q, self.p))

        print(self.n, self.d, self.q, "DONE")

    def show(self, path, label):
        tosave = np.asarray([self.sigmas, self.p_err, self.p_check])
        np.savetxt("%s/NTRU_sim_%s_%s_n%i_q%i_p%i_d%i_NMAX%i.txt" % (
            path, label, self.prefix, self.n, self.q, self.p, self.d, self.NUM_MAX), tosave)

        plt.figure("checking_%s_n%i_q%i_p%i" % (label, self.n, self.q, self.p))
        plt.plot(self.sigmas, self.p_err, label="p_err")
        plt.plot(self.sigmas, self.p_check, label="p_check")
        plt.legend()
        plt.show()
        # plt.savefig("%s/checking_n%i_q%i_p%i.pdf" % (path, self.n, self.q, self.p))

        print(self.n, self.d, self.q, "DONE")

        return 0

# In[160]:

cpu_count=multiprocessing.cpu_count()

Ns=[7,11,17,37]#, 97]
Ds=[n//3 for n in Ns]
Qs=[4,8,16,32]#,64]
Ps=[[3],[5],[7],[3,5]]#[[3]]*len(Ns)#
NUM_MAX=int(1e5)


sigmas=np.linspace(1e-7,0.2,100)

hinv=True
p3=True



if hinv:

    if p3:
        loadfolder = "pNTRU_lats_hinv_p3"
        savefolder = "results_hinv_p3"

        Ps=[[3]]*len(Ns)


    else:
        loadfolder = "pNTRU_lats_hinv"
        savefolder = "results_hinv"

else:

    if p3:
        loadfolder = "pNTRU_lats_p3"
        savefolder = "results_p3"

        Ps=[[3]]*len(Ns)

    else:

        loadfolder = "pNTRU_lats"
        savefolder = "results"



if __name__ == '__main__':

    for k in range(len(Ns)-1):

        n, q, d, p_list = Ns[k], Qs[k], Ds[k], Ps[k]

        if hinv:
            worker_NTRU = Cload_and_sim(sigmas, n, q, p_list, d, "opt", loadfolder, NUM_MAX, precision=12, inv=hinv)

        #worker_BAB = Cload_and_sim(sigmas, n, q, p_list, d, "opt", loadfolder, NUM_MAX, precision=12, inv=hinv)



        with Pool(cpu_count-3) as p:

            if hinv:
                data_NTRU=p.map(worker_NTRU.evaluate, list(range(len(sigmas))))

            #data_BAB = p.map(worker_BAB.BabaiX_evaluate, list(range(len(sigmas))))

        for i in range(len(sigmas)):
            if hinv:
                worker_NTRU.p_err[i], worker_NTRU.p_check[i] = data_NTRU[i][0], data_NTRU[i][1]
            #worker_BAB.p_err[i], worker_BAB.p_check[i] = data_BAB[i][0], data_BAB[i][1]

        if hinv:
            worker_NTRU.save(savefolder, "NTRU")
        #worker_BAB.save(savefolder, "BAB")







