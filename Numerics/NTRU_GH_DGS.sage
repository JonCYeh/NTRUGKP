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





#--------------------------------------------------------------------------------------------#


import numpy as np

def T_Phi(Phi):
    n=len(Phi)
    T=np.zeros((n,n))
    T[1:,:-1]=np.eye(n-1)
    T[:,-1]=-Phi

    return T

def C_Phi(f, Phi):
    n=len(f)

    T=T_Phi(Phi)

    TEMP=np.zeros((n,n))
    TEMP[0]=f
    for i in range(1,n):
        TEMP[i]=T@TEMP[i-1]

    return TEMP

def A_Phi(f, Phi):
    n=len(f)

    Tinv=np.linalg.inv(T_Phi(Phi))

    TEMP=np.zeros((n,n))
    TEMP[0]=f

    for i in range(n-1):
        TEMP[i+1,:]=Tinv@TEMP[i,:]

    return TEMP

def sigma_Phi(Phi):
    ID=np.zeros(len(phi))
    ID[0]=1

    return A_Phi(ID, Phi)

def NTRU_ABAS(h, Phi, q):
    n=len(h)
    return np.block([[np.eye(n), A_Phi(h, Phi)],[np.zeros((n,n)), q*np.eye(n)]])



from sage.modules.free_module_integer import IntegerLattice

def A_pNTRU_len(h,Phi,q):
    pub_bas = NTRU_ABAS(h, Phi, q)
    Mp_bas=(pub_bas)
    L = IntegerLattice(Mp_bas)

    L.HKZ(algorithm="NTL", proof=True)

    SVP=min(v.norm().n() for v in L.reduced_basis)

    return SVP


def invertmodprime(f, p, n, Phi):
    Coeff_poly=list(Phi)+[1]
    Phi_poly=Zx(Coeff_poly)
    T = Zx.change_ring(Integers(p)).quotient(x ^ n - 1)
    return Zx(lift(1 / T(f)))


def invertmodpowerof2(f, q, n, Phi):
    assert q.is_power_of(2)
    g = invertmodprime(f, 2, n)
    while True:
        r = balancedmod(convolution(g, f, n), q, n)
        if r == 1: return g
        g = balancedmod(convolution(g, 2 - r, n), q, n)


def QuotientRing(Phi):
    Coeff_poly=list(Phi)+[1]
    Phi_poly=Zx(Coeff_poly)

    return Zx.quotient(Phi_poly)

def QuotientRing_q(Phi,q):

    QuotientPhi=QuotientRing(Phi)

    return QuotientPhi.quotient(q)

def coeff(f, n):
    return np.asarray([f[i] for i in range(n)])


from sage.stats.distributions.discrete_gaussian_polynomial import DiscreteGaussianDistributionPolynomialSampler

def DGS_sampler(n,q,Phi, sigma):
    QuotientPhi=QuotientRing(Phi)
    gs = DiscreteGaussianDistributionPolynomialSampler(QuotientPhi, n, sigma)

    return gs


def DGS_keygen(n,q,Phi, c, num):

    q_pow2=(np.log2(q)==np.round(np.log2(q)))
    n_pow2=(np.log2(n)==np.round(np.log2(n)))
    #print("Success prob", np.float(1-n/q))

    #print("n is power of 2",n_pow2)
    #print("q is power of 2",q_pow2)
    Sampler=DGS_sampler(n,q,Phi,n**c*np.sqrt(q) )

    Samples=[]
    for i in range(num):

        for _ in range(int(1e7)):
            try:
                f =Sampler()
                g =Sampler()
                if q_pow2:
                    fq = invertmodpowerof2(f, q, n, Phi)
                    gq = invertmodpowerof2(g, q, n, Phi)
                else:
                    fq = invertmodprime(f, q, n, Phi)
                    gq = invertmodprime(g, q, n, Phi)


                break

            except:
                pass


        h=balancedmod(fq*g , q, n)
        #print('h',h)

        f_vec=coeff(f,n)
        #fq_vec=coeff(fq,n)
        g_vec=coeff(g,n)
        #gq_vec=coeff(gq,n)
        h_vec=coeff(h,n)

        #Cf=C_Phi(f_vec, Phi)
        #Cfq=C_Phi(fq_vec, Phi)
        #Cg=C_Phi(g_vec, Phi)
        #Cgq=C_Phi(gq_vec, Phi)

        #h_vec= rbalancedmod_ar( (Cfq@Cg)[0], q)


        #print('hvec', h_vec)

        Samples+=[(h_vec, (f_vec, g_vec))]

    return Samples

def n_lambda1(k_nmin, k_nmax, q, c, NUM, n_pow2):

    distr=[]

    for kn in range(k_nmin, k_nmax+1):

        if n_pow2:
            n=2**kn
            print(n)
        else:
            n=kn
        Phi=np.zeros(n)
        Phi[0]=1
        samples=DGS_keygen(n,q,Phi, c, NUM)

        for S in samples:
            h,fg=S
            #print(h)
            lambda1=A_pNTRU_len(h,Phi,q)

            distr+=[(n, lambda1)]


    folder = "Sampling_DGS_new"

    np.savetxt('%s/NTRU_lambda_samples_DGS_c%.2f_HKZlambda_q%i.txt'%(folder,c,q), np.asarray(distr))

    nmax=distr[-1][0]
    gp=scatter_plot(distr, alpha=0.5, markersize=10, axes_labels=[r'$n$', r'$\lambda_1$'])
    gp=gp+plot(sqrt(x*q/(pi*e)), (x,0,nmax))
    gp=gp+plot(sqrt(0.28*x), (x,0,nmax), color='red')
    gp=gp+plot(q^(1/(2)),(x, 0,nmax), color='green')



    gp.save('%s/NTRU_lambda_samples_DGS_c%.2f_HKZlambda_q%i.pdf'%(folder,c,q))

    return distr


k_nmin, k_nmax=2,4
for q in [ 167, 367,509, 1021, 2027]:

    print("q=", q)
    n_lambda1(k_nmin, k_nmax, q, 0, 100, 1)
