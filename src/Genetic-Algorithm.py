import bisect
import math
import numpy as np
import matplotlib.pyplot as plt

def Initialisation(P, S, B):
    mems = np.zeros((P, B), dtype=int)
    for i in range(P):
        for j in range(B-1):
            mems[i][j] = np.random.randint(S-int(mems[i].sum())+1)
        mems[i][B-1] = S - int(mems[i].sum())
    return mems

def Fitness(mems, mem):
    score = 0.0
    P = len(mems)
    B = len(mem)
    for i in range(P):
        rival = mems[i]
        tmp=0
        for j in range(B):
            if(mem[j] > rival[j]):
                tmp += 1
            elif(mem[j] < rival[j]):
                tmp -= 1
        if(tmp>0):
            score += 1
    return score/P

global GMEMS
def iter(mem):
    return Fitness(GMEMS, mem)

def NewFitness(mems, mem):
    score = 0.0
    P = len(mems)
    B = len(mem)
    for i in range(P):
        rival = mems[i].copy()
        m = mem.copy()
        tmp=0
        for j in range(B):
            if(m[j] > rival[j]):
                tmp += 1
                if(j<B-1):
                    a = m[j]-rival[j]
                    aa = int(1.0*a/(B-j-1))
                    a = a-aa
                    for k in range(j+1, B):
                        m[k] += min(a, 1) + aa
                        a-=1
            elif(m[j] < rival[j]):
                tmp -= 1
                if(j<B-1):
                    a = rival[j]-m[j]
                    aa = int(1.0*a/(B-j-1))
                    a = a-aa
                    for k in range(j+1, B):
                        rival[k] += min(a, 1) + aa
                        a-=1
        if(tmp>0):
            score += 1
    return score/P

def Newiter(mem):
    return NewFitness(GMEMS, mem)

def mySort(mems, iter):
    global GMEMS
    GMEMS = mems
    mems = mems.tolist()
    mems.sort(key=iter)
    mems = np.array(mems)
    return mems

def Tournament_selection(mems, k, Fitness):
    P = len(mems)
    nmems = []
    for i in np.random.permutation(P)[:k]:
        nmems += [mems[i]]
    m0 = 0
    m1 = 1
    if(Fitness(mems, nmems[m0])<Fitness(mems, nmems[m1])):
        m0, m1 = m1, m0
    for i in range(2, k):
        if(Fitness(mems, nmems[m1])<Fitness(mems, nmems[i])):
            m1 = i
        if(Fitness(mems, nmems[m0])<Fitness(mems, nmems[m1])):
            m0, m1 = m1, m0
    return [nmems[m0], nmems[m1]]

def Proportionate_selection(mems, Fitness):
    P = len(mems)
    k = 0
    for i in range(P):
        k += int(Fitness(mems, mems[i]) * P + 1)
    r = np.random.randint(k) + 1
    m0 = 0
    while(r>0):
        r -= int(Fitness(mems, mems[i]) * P + 1)
        m0 += 1
    m0 -= 1
    r = np.random.randint(k) + 1
    m1 = 0
    while(r>0):
        r -= int(Fitness(mems, mems[i]) * P + 1)
        m1 += 1
    m1 -= 1
    return [mems[m0], mems[m1]]

def Swap_mutation(mem, prob):
    if(np.random.rand() > prob):
        return mem
    B = len(mem)
    p = np.random.permutation(B)[:2]
    mem[p[0]], mem[p[1]] = mem[p[1]], mem[p[0]]
    return mem

def Boundary_mutation(mem, prob):
    if(np.random.rand() > prob):
        return mem
    B = len(mem)
    p = np.random.randint(B)
    a = mem[p]
    mem[p] = 0
    for i in range(a):
        p = np.random.randint(B)
        mem[p] += 1
    return mem

def survival_elitism(mems, child, ELITISM):
    P = len(mems)
    j = np.random.randint(int(P*(1-ELITISM)))
    mems[j] = child
    return mems

# Variables initialisation:

S = 20 # number of soldiers
B = 4 # number of battles
P = 50 # number of population
ELITISM = 0.1 # elitism probability
ITER = 100
K = 5 # number of select for Tournament selection
Mprob = 1 # mutation probability
data = [] # used for analyze
Mmems = Initialisation(P, S, B) # population initialisation

# ------------------------------------ Part A -----------------------------------

# Genetic-algorithm:

mems = Mmems.copy()
mems = mySort(mems, iter)
data = []
data += [mems.copy()]

for _ in range(ITER):
    parents = Tournament_selection(mems, K, Fitness)
    childs = parents.copy()
    childs[0] = Swap_mutation(childs[0], Mprob)
    childs[1] = Swap_mutation(childs[1], Mprob)
    survival_elitism(mems, childs[0], ELITISM)
    survival_elitism(mems, childs[1], ELITISM)
    mems = mySort(mems, iter)
    data += [mems.copy()]


# Analyze:

maxFit = [] # max fitness
avgFits = [] # avg of k-max fitness
l = len(data) # last result
print("Part A")

for i in range(l):
    maxFit += [Fitness(data[i], data[i][P-1])]
    print(data[i][P-1])
    tmp = 0
    for j in range(P):
        tmp += Fitness(data[i], data[i][j])
    avgFits += [1.0*tmp/P]

plt.figure(1)

plt.subplot(2, 1, 1)
plt.plot(maxFit, '.-')
plt.title('Fitness Analyze Part A')
plt.ylabel('Max Fitness')

plt.subplot(2, 1, 2)
plt.plot(avgFits, '.-')
plt.xlabel('Generations')
plt.ylabel('Avg Fitness')

# ------------------------------------ Part B -----------------------------------

# Genetic-algorithm:

mems = Mmems.copy()
mems = mySort(mems, iter)
data = []
data += [mems.copy()]
print("Part B")

for _ in range(ITER):
    parents = Tournament_selection(mems, K, Fitness)
    childs = parents.copy()
    childs[0] = Boundary_mutation(childs[0], Mprob)
    childs[1] = Boundary_mutation(childs[1], Mprob)
    survival_elitism(mems, childs[0], ELITISM)
    survival_elitism(mems, childs[1], ELITISM)
    mems = mySort(mems, iter)
    data += [mems.copy()]


# Analyze:

maxFit = [] # max fitness
avgFits = [] # avg of k-max fitness
l = len(data) # last result

for i in range(l):
    maxFit += [Fitness(data[i], data[i][P-1])]
    print(data[i][P-1])
    tmp = 0
    for j in range(P):
        tmp += Fitness(data[i], data[i][j])
    avgFits += [1.0*tmp/P]

plt.figure(2)

plt.subplot(2, 1, 1)
plt.plot(maxFit, '.-')
plt.title('Fitness Analyze Part B')
plt.ylabel('Max Fitness')

plt.subplot(2, 1, 2)
plt.plot(avgFits, '.-')
plt.xlabel('Generations')
plt.ylabel('Avg Fitness')

# ------------------------------------ Part C -----------------------------------

# Genetic-algorithm:

mems = Mmems.copy()
mems = mySort(mems, iter)
data = []
data += [mems.copy()]
print("Part C")

for _ in range(ITER):
    parents = Proportionate_selection(mems, Fitness)
    childs = parents.copy()
    childs[0] = Swap_mutation(childs[0], Mprob)
    childs[1] = Swap_mutation(childs[1], Mprob)
    survival_elitism(mems, childs[0], ELITISM)
    survival_elitism(mems, childs[1], ELITISM)
    mems = mySort(mems, iter)
    data += [mems.copy()]


# Analyze:

maxFit = [] # max fitness
avgFits = [] # avg of k-max fitness
l = len(data) # last result

for i in range(l):
    maxFit += [Fitness(data[i], data[i][P-1])]
    print(data[i][P-1])
    tmp = 0
    for j in range(P):
        tmp += Fitness(data[i], data[i][j])
    avgFits += [1.0*tmp/P]

plt.figure(3)

plt.subplot(2, 1, 1)
plt.plot(maxFit, '.-')
plt.title('Fitness Analyze Part C')
plt.ylabel('Max Fitness')

plt.subplot(2, 1, 2)
plt.plot(avgFits, '.-')
plt.xlabel('Generations')
plt.ylabel('Avg Fitness')

# ------------------------------------ Part D -----------------------------------

# Genetic-algorithm:

mems = Mmems.copy()
mems = mySort(mems, Newiter)
data = []
data += [mems.copy()]

for _ in range(ITER):
    parents = Tournament_selection(mems, K, NewFitness)
    childs = parents.copy()
    childs[0] = Swap_mutation(childs[0], Mprob)
    childs[1] = Swap_mutation(childs[1], Mprob)
    survival_elitism(mems, childs[0], ELITISM)
    survival_elitism(mems, childs[1], ELITISM)
    mems = mySort(mems, Newiter)
    data += [mems.copy()]


# Analyze:

maxFit = [] # max fitness
avgFits = [] # avg of k-max fitness
l = len(data) # last result
print("Part D")

for i in range(l):
    maxFit += [NewFitness(data[i], data[i][P-1])]
    print(data[i][P-1])
    tmp = 0
    for j in range(P):
        tmp += NewFitness(data[i], data[i][j])
    avgFits += [1.0*tmp/P]

plt.figure(4)

plt.subplot(2, 1, 1)
plt.plot(maxFit, '.-')
plt.title('Fitness Analyze Part D')
plt.ylabel('Max Fitness')

plt.subplot(2, 1, 2)
plt.plot(avgFits, '.-')
plt.xlabel('Generations')
plt.ylabel('Avg Fitness')

plt.show()
