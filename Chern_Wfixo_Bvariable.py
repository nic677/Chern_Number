import numpy as np
import sys






W = float(sys.argv[1])
N = int(sys.argv[2])
BS = np.linspace(-2.5,-1.5,int(sys.argv[3]))
t = 1 
NAV = int(sys.argv[4])


def Hamil(N, B, W): 
    H = np.zeros((2*N**2, 2*N**2), dtype=np.complex128)
    for i in range(N):
        
            #1st orbital
        H[i*N + (np.arange(N)+1)%N, i*N + np.arange(N)] = t
        H[i*N + (np.arange(N)-1)%N, i*N + np.arange(N)] = t
        H[(N*(i+1))%(N**2) + np.arange(N), i*N + np.arange(N)] = -t
        H[(N*(i-1))%(N**2) + np.arange(N), i*N + np.arange(N)] = -t
            
            #2nd orbital
        H[N**2 + i*N + (np.arange(N)+1)%N,N**2 +  i*N + np.arange(N)] = -t
        H[N**2 + i*N + (np.arange(N)-1)%N, N**2 + i*N + np.arange(N)] = -t
        H[N**2 + (N*(i+1))%(N**2) + np.arange(N), N**2 + i*N + np.arange(N)] = t
        H[N**2 + (N*(i-1))%(N**2) + np.arange(N),N**2 +  i*N + np.arange(N)] = t
            
            #NNN hoppings
            
        H[(N*(i+1))%(N**2) + (np.arange(N)+1)%(N), N**2 + i*N + np.arange(N)] = 1./2.
        H[(N*(i-1))%(N**2) + (np.arange(N)-1)%(N), N**2 + i*N + np.arange(N)] = 1./2.
        H[(N*(i+1))%(N**2) + (np.arange(N)-1)%(N), N**2 + i*N + np.arange(N)] = -1./2.
        H[(N*(i-1))%(N**2) + (np.arange(N)+1)%(N), N**2 + i*N + np.arange(N)] = -1./2.
            
        H[N**2 + (N*(i+1))%(N**2) + (np.arange(N)+1)%(N),  i*N + np.arange(N)] = 1./2.
        H[N**2 +(N*(i-1))%(N**2) + (np.arange(N)-1)%(N),  i*N + np.arange(N)] = 1./2.
        H[N**2 +(N*(i+1))%(N**2) + (np.arange(N)-1)%(N),  i*N + np.arange(N)] = -1./2.
        H[N**2 +(N*(i-1))%(N**2) + (np.arange(N)+1)%(N),  i*N + np.arange(N)] = -1./2.
            
            #B parameter
            
        H[N**2 + i*N + np.arange(N), i*N + np.arange(N)] = -1j
        H[i*N + np.arange(N), N**2 + i*N + np.arange(N)] = 1j
        H[N**2 + i*N + (np.arange(N)+1)%N, i*N +np.arange(N)] = -1.j*(B+1.)/4.
        H[N**2 + i*N + (np.arange(N)-1)%N, i*N +np.arange(N)] = -1.j*(B+1.)/4.
            
        H[N**2 + (N*(i+1))%(N**2) + np.arange(N), i*N +np.arange(N)] = -1j*(B+1)/4.
        H[N**2 + (N*(i-1))%(N**2) + np.arange(N), i*N +np.arange(N)] = -1j*(B+1)/4.
            
        H[N + (np.arange(N)+1)%N,N**2+ i*N +np.arange(N)] = 1j*(B+1)/4.
        H[ i*N + (np.arange(N)-1)%N, N**2+i*N +np.arange(N)] = 1j*(B+1)/4.
            
        H[(N*(i+1))%(N**2) + np.arange(N),N**2+ i*N +np.arange(N)] = 1j*(B+1)/4.
        H[ (N*(i-1))%(N**2) + np.arange(N), N**2+i*N +np.arange(N)] = 1j*(B+1)/4.
            
            
        #Anderson Disorder
        H[np.arange(2*N**2), np.arange(2*N**2)] = W*(np.random.random(2*N**2)- 0.5)
    return H


def get_st(i, N):
    return (float(i)%float(N), float(i)//float(N))

def exp_vec(q,q1,N):
    x = np.array([np.exp(2*np.pi*1j*((q[0] - q1[0])*get_st(i,N)[0] + (q[1] - q1[1])*get_st(i,N)[1])) for i in range(N**2)])

    return x
def C(q,q1,N,eigenVectors):    
        resultado = np.dot((exp_vec(q,q1,N) * ((np.conj(eigenVectors[:N**2,:N**2])).T)), eigenVectors[:N**2,:N**2]) + np.dot((exp_vec(q,q1,N) * ((np.conj(eigenVectors[N**2:,:N**2])).T)), eigenVectors[N**2:,:N**2])  
#         print(resultado.shape())
        return resultado
    
    
def Ca(B, W):
    eigenVectors = np.linalg.eigh(Hamil(N, B, W))[1]

    q0 = np.array([0,0])
    q1 = np.array([1/N,0])
    q2 = np.array([1/N,1/N])
    q3 = np.array([0,1/N])

    resultado = np.dot(C(q0,q1,N, eigenVectors), np.dot(C(q1,q2, N, eigenVectors), np.dot(C(q2,q3,N, eigenVectors),C(q3,q0,N, eigenVectors))))
    return -np.sum(np.angle(np.linalg.eigvals(resultado)))/2/np.pi
    
Array = np.zeros(len(BS))
for i in range(NAV):
    for j in range(len(BS)):
        Array[j] += Ca(BS[j], W)/NAV
np.savetxt("Chern_W" + str(W) + "N"+str(N)+"Nav"+str(NAV)+".txt",Array)

