Zx.< x > = ZZ[]

def convolution(f, g, n):
    return (f * g) % (x ^ n - 1)


def balancedmod(f, q, n):
    g = list(((f[i] + q // 2) % q) - q // 2 for i in range(n))
    return Zx(g)


def balancedmod_ar(ar, q):
    return (ar + q / 2) % q - q / 2


def rbalancedmod_ar(ar, q):
    return (ar + q // 2) % q - q // 2


def randomdpoly(n, d):
    # print("rpoly",n,d)

    assert d <= n
    result = n * [0]
    for j in range(d):
        while True:
            r = randrange(n)
            # print(r)
            if not result[r]: break
        result[r] = 1 - 2 * randrange(2)
    print(result)
    return Zx(result)


def Tpoly(n, dp, dm):
    # print("rpoly",n,d)
    assert (dp + dm <= n)
    s = [1] * dp + [-1] * dm + [0] * (n - dp - dm)
    shuffle(s)

    return Zx(s)


def randomdpolyq(n, d, q):
    assert d <= n
    assert q % 2 == 0

    result = n * [0]

    for j in range(d):
        result[j] = q / 2 - randrange(q) - 1

    p = Permutations(len(result)).random_element()
    result_rand = [result[p[i] - 1] for i in range(len(result))]
    out = Zx(result_rand)

    return out


def invertmodprime(f, p, n):
    T = Zx.change_ring(Integers(p)).quotient(x ^ n - 1)
    return Zx(lift(1 / T(f)))


def invertmodpowerof2(f, q, n):
    assert q.is_power_of(2)
    g = invertmodprime(f, 2, n)
    while True:
        r = balancedmod(convolution(g, f, n), q, n)
        if r == 1: return g
        g = balancedmod(convolution(g, 2 - r, n), q, n)


def keypair(n, d, q):
    while True:
        try:
            f = randomdpoly(n, d)
            f3 = invertmodprime(f, 3, n)
            fq = invertmodpowerof2(f, q, n)
            break
        except:
            pass
    g = randomdpoly(n, d)
    # print(g)
    publickey = balancedmod(3 * convolution(fq, g, n), q, n)
    secretkey = f, f3
    return publickey, secretkey


def randommessage(n):
    result = list(randrange(3) - 1 for j in range(n))
    return Zx(result)


def encrypt(message, publickey, n, d):
    r = randomdpoly(n, d)
    return balancedmod(convolution(publickey, r, n) + message, q, n)


def decrypt(ciphertext, secretkey):
    f, f3 = secretkey
    a = balancedmod(convolution(ciphertext, f, n), q, n)
    return balancedmod(convolution(a, f3, n), 3, n)


import numpy as np


def T(f):
    temp = np.zeros((len(f), len(f)))
    for i in range(len(f)):
        temp[i] = np.roll(f, i)

    return temp


def That(f):
    temp = np.zeros((len(f), len(f)))
    for i in range(len(f)):
        temp[i] = np.roll(f, -i)

    return temp


def sigma(f):
    temp = np.zeros(len(f))
    temp[0] = f[0]
    temp[1:] = f[1:][::-1]

    return temp


def Tsigma(f):
    temp = np.zeros((len(f), len(f)))
    for i in range(len(f)):
        temp[i] = sigma(np.roll(f, i))

    return temp


def sec_basis(f, g, q, n):
    f_sigma = Tsigma(f)
    Tg = T(g)
    return np.block([[f_sigma, Tg, np.zeros((n, n)), q * np.eye(n)]])


def Xsec_basis(f, g, q, n):
    Tf_sigma = Tsigma(f)
    Tg = T(g)
    return np.block([[Tg, Tf_sigma]])


from tqdm import tqdm


def mat_keypair(n, d, q, p):
    for i in tqdm(range(int(1e7))):
        try:
            f = 1 + p * Tpoly(n, d, d)
            fq = invertmodpowerof2(f, q, n)
            g = p * Tpoly(n, d+1, d)
            publickey = balancedmod(convolution(fq, g, n), q, n)

            public_inv = invertmodpowerof2(publickey, q, n)
            print("SUCCESS")
            break
        except:
            pass

    h = np.asarray([publickey[i] for i in range(n)])
    h_inv = np.asarray([public_inv[i] for i in range(n)])
    f_vec = np.asarray([f[i] for i in range(n)])
    g_vec = np.asarray([g[i] for i in range(n)])

    pk = That(h)
    sk = Tsigma(f_vec)
    pk_inv = Tsigma(h_inv)

    return h, h_inv, f_vec, g_vec


def mat_keypair_ninv(n, d, q, p):
    for i in range(int(1e7)):
        try:
            f = 1 + p * Tpoly(n, d, d)  # f
            g = p * Tpoly(n, d, d)
            # print(f,g)

            fq = invertmodpowerof2(f, q, n)

            publickey = balancedmod(convolution(fq, g, n), q, n)
            # print(fq)

            # public_inv=invertmodpowerof2(publickey,q,n)
            print("SUCCESS")
            break
        except:
            pass

    h = np.asarray([publickey[i] for i in range(n)])
    # h_inv=np.asarray([public_inv[i] for i in range(n)])
    f_vec = np.asarray([f[i] for i in range(n)])
    g_vec = np.asarray([g[i] for i in range(n)])

    pk = That(h)
    sk = Tsigma(f_vec)
    # pk_inv=Tsigma(h_inv)

    return h, f_vec, g_vec


def pub_basis(pk, q, n, l):
    return np.sqrt(l / q) * np.block([[np.eye(n), pk], [np.zeros((n, n)), q * np.eye(n)]])


def randomstring(n):
    return 1 - np.random.randint(3, size=n)


def cipher(m, r, pk, n, q):
    return balancedmod_ar(m + pk @ r, q)


def Jsym(n):
    J1 = np.asarray([[0, 1], [-1, 0]])
    J = np.kron(J1, np.eye(n))
    return J


def NTRUlattice(publickey, n, d, q, p):
    # n,d,q=n,d,q
    # publickey,secretkey = keypair(n,d,q)
    recipp = lift(1 / Integers(q)(p))
    publickeyoverp = balancedmod(recipp * publickey, q, n)

    M = matrix(2 * n)
    for i in range(n, 2 * n):
        M[i, i] = q
    for i in range(n):
        M[i, i] = 1
        c = convolution(x ^ (n - i), publickeyoverp, n)
        for j in range(n):
            M[i, j + n] = c[j]

    return M


from sage.modules.free_module_integer import IntegerLattice


def NTRU_len(publickey, n, d, q, p):
    M = NTRUlattice(publickey, n, d, q, p)
    L = IntegerLattice(M)

    SVP = L.shortest_vector()  # algorithm="pari")

    return SVP.norm().n()


def NTRU_len(publickey, n, d, q, p):
    M = NTRUlattice(publickey, n, d, q, p)
    L = IntegerLattice(M)

    L.BKZ(block_size=n)
    # SVP=L.shortest_vector()#algorithm="pari")

    return min(v.norm().n() for v in L.reduced_basis)  # SVP.norm().n()

def pNTRU_len(h, n,q):
    pk = That(h)
    pub_bas = np.block([[np.eye(n), pk], [np.zeros((n, n)), q * np.eye(n)]])
    Mp_bas=(pub_bas)
    L = IntegerLattice(Mp_bas)

    L.HKZ(algorithm="NTL", proof=True)
    # SVP=min(np.linalg.norm(v) for v in s_bas)
    # L.LLL()
    SVP=min(v.norm().n() for v in L.reduced_basis)
    #try:
    #SVP = L.shortest_vector().norm().n()
    #except:
    #SVP=L.shortest_vector().norm().n()
    # print(SVP)

    return SVP


def sNTRU_len(f, g, n,q):
    f_sigma = Tsigma(f)
    Tg = T(g)
    s_bas=np.concatenate([np.block([[f_sigma, Tg]]),np.eye(2*n)])
    Ms_bas=Matrix(s_bas)

    L = IntegerLattice(Ms_bas)

    L.HKZ(algorithm="NTL", proof=True)
    #SVP=min(np.linalg.norm(v) for v in s_bas)
    # L.LLL()
    SVP=min(v.norm().n() for v in L.reduced_basis)
    #try:
    #SVP = L.shortest_vector().norm().n()
    #except:
    #SVP=L.shortest_vector().norm().n()
    # print(SVP)

    return SVP


def pub_NTRU_GS(h, n,q):
    K=Integers(q)
    pk = That(h)
    pub_bas = np.block([[np.eye(n), pk], [np.zeros((n, n)), q * np.eye(n)]])
    Mat_pb = Matrix(pub_bas)
    L = IntegerLattice(Mat_pb)
    L.LLL()
    # print("LLL basis\n")
    # print(L.reduced_basis)
    # print("GS\n")
    # print(L.reduced_basis.gram_schmidt())
    # print("minlen GS \n")
    # print(min(v.norm().n() for v in L.reduced_basis.gram_schmidt()))
    min_gs = min(v.norm().n() for v in L.reduced_basis.gram_schmidt())

    return min_gs


def s_NTRU_GS(f, g, n, q):
    K=Integers(q)
    f_sigma = Tsigma(f)
    Tg = T(g)
    s_bas = np.block([[f_sigma, Tg]])#, np.zeros((n, n)), q * np.eye(n)]])

    Mat_sb = Matrix(K,s_bas)
    L = IntegerLattice(Mat_sb)
    L.LLL()
    # print("LLL basis\n")
    # print(L.reduced_basis)
    # print("GS\n")
    # print(L.reduced_basis.gram_schmidt())
    # print("minlen GS \n")
    # print(min(v.norm().n() for v in L.reduced_basis.gram_schmidt()))
    min_gs = min(v.norm().n() for v in L.reduced_basis.gram_schmidt())

    return min_gs


def LLL_save(path, M):
    Mat_M = Matrix(M)
    L = IntegerLattice(Mat_M)
    L.LLL()
    LLL_bas = np.asarray(L.reduced_basis)

    np.savetxt(path, LLL_bas)

    return

def HKZ_save(path, M):
    Mat_M = Matrix(M)
    L = IntegerLattice(Mat_M)
    L.HKZ(algorithm="NTL", proof=True)
    HKZ_bas = np.asarray(L.reduced_basis)

    np.savetxt(path, HKZ_bas)

    return

def Xsec_LLL_save(f, g, q, n, path):
    X_sec = Xsec_basis(f, g, q, n)
    LLL_save(path, X_sec)

    return

def Xsec_HKZ_save(f, g, q, n, path):
    X_sec = Xsec_basis(f, g, q, n)
    HKZ_save(path, X_sec)

    return



#--------------------------------------------------------------------------------------------#


NUM=100

hinv=False
p3=True


Ns=[7,11,17,37]#, 97]
Ds=[n//3 for n in Ns]
Qs=[4,8,16,32]#,64]

if p3:
    Ps=[3]*len(Ns)
else:
    Ps=[3,5,7,15]#,21]

#Ds=[5,7,11,29,53]
opt_len=[0]*len(Ns)
#len_plot=[]


for i in range(len(Ns)-1):
    q=Qs[i]

    p=Ps[i]
    n=Ns[i]
    d=Ds[i]
    #print("n,d=", n,d)
    temp=[]
    temp_len=[0]*NUM
    for _ in tqdm(range(NUM)):

        if hinv:
            h,h_inv,f_vec, g_vec=mat_keypair(n,d,q,p)
            tl=pNTRU_len(h,n,q)#NTRU_len(h,n,d,q,p)
            temp_len[_]=tl
            temp+=[[h,h_inv, f_vec, g_vec, tl*np.ones(len(f_vec))]]

        else:
            h,f_vec, g_vec=mat_keypair_ninv(n,d,q,p)
            tl=pNTRU_len(h,n,q)#NTRU_len(h,n,d,q,p)
            temp_len[_]=tl
            temp+=[[h, f_vec, g_vec, tl*np.ones(len(f_vec))]]


    indmax=np.argmax(temp_len)
    opt_len[i]=temp_len[indmax]
    optimal=np.asarray(temp[indmax])
    #print(optimal[0].shape)
    #print("opt_l=%.3f"%(temp_len[indmax]))


    if hinv:
        if p3:
            folder="pNTRU_lats_hinv_p3"
            np.savetxt("%s/NTRU_opt_n%i_q%i_p%i_d%i.txt"%(folder, n,q,p,d), optimal,comments="#sLlambda_1=%.2f"%(temp_len[indmax]))
            Xsec_HKZ_save(optimal[2], optimal[3], q,n,  "%s/NTRU_HKZ_n%i_q%i_p%i_d%i.txt"%(folder, n,q,p,d))

        else:
            folder="pNTRU_lats_hinv"
            np.savetxt("%s/NTRU_opt_n%i_q%i_p%i_d%i.txt"%(folder, n,q,p,d), optimal,comments="#sLlambda_1=%.2f"%(temp_len[indmax]))
            Xsec_HKZ_save(optimal[2], optimal[3], q,n,  "%s/NTRU_HKZ_n%i_q%i_p%i_d%i.txt"%(folder, n,q,p,d))

    elif not hinv:
        if p3:
            folder="pNTRU_lats_p3"
            np.savetxt("%s/NTRU_opt_n%i_q%i_p%i_d%i.txt"%(folder, n,q,p,d), optimal,comments="#sLlambda_1=%.2f"%(temp_len[indmax]))
            Xsec_HKZ_save(optimal[1], optimal[2], q,n,  "%s/NTRU_HKZ_n%i_q%i_p%i_d%i.txt"%(folder, n,q,p,d))

        else:
            folder="pNTRU_lats"
            np.savetxt("%s/NTRU_opt_n%i_q%i_p%i_d%i.txt"%(folder, n,q,p,d), optimal,comments="#sLlambda_1=%.2f"%(temp_len[indmax]))
            Xsec_HKZ_save(optimal[1], optimal[2], q,n,  "%s/NTRU_HKZ_n%i_q%i_p%i_d%i.txt"%(folder, n,q,p,d))


    print("n,d=", n,d, "DONE")


import matplotlib.pyplot as plt
plt.plot(Ns, opt_len)

plt.savefig("%s/sample_scale.pdf"%(folder))


opt_params=np.asarray([[Ns[i], Ds[i], opt_len[i]] for i in range(len(Ns))])
np.savetxt("%s/NTRU_parameters_q%i_p%i_NUM%i.txt"%(folder, q,p,NUM), opt_params)

print("done")
