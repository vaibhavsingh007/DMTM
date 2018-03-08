def supportCount(itemset, T):    # returns count of itemset in T
    count = 0

    # Increment count if itemset in t
    for t in T:
        is_in_t = len(itemset) > 0
        for i in itemset:
            is_in_t = is_in_t and i in t
        if is_in_t:
            count += 1
    
    return count

support = lambda count, T: float(count) / len(T)
sdcSatisfied = lambda min_count, max_count, T, sdc: abs(support(max_count, T) - support(min_count, T)) <= sdc

def notTogetherSatisfied(not_together, c):   # 2D and 1D respectively
    for nt in not_together:
        valid = False
        for i in nt:
            if i not in c:
                valid = True
        if not valid:
            return False
    return True

def mustHaveSatisfied(must_have, c):    # 2D and 1D respectively
    for i in must_have:
        if i in c:
            return True
    return False

def initPass(M, T, MS):
    l = []
    l_count = []    # to memoize the support count of l

    for i,m in enumerate(M):
        sup_count = supportCount([m], T)

        if support(sup_count, T) >= MS[m]:
            # Add m to l, and all subsequent j"s if sup(j) >= MS[m]
            l.append(m)
            l_count.append(sup_count)

            for j in M[i + 1:]:
                sup_count = supportCount([j], T)

                if support(sup_count, T) >= MS[m]:
                    l.append(j)
                    l_count.append(sup_count)
            break
    return l, l_count

def level2_candidate_gen(L, L_counts, T, MS, sdc, not_together, must_have):
    C = []

    for i,l in enumerate(L):
        if support(L_counts[i], T) >= MS[l]:

            for i1,j in enumerate(L[i + 1:]):
                i_ = i1 + i + 1     # Managing the indices to access correct counts
                if support(L_counts[i_], T) >= MS[l]:
                    # Check constraints
                    if sdcSatisfied(L_counts[i], L_counts[i_], T, sdc):
                        C.append([l,j])
    return C

def kLess1Subsets(c):
    subsets = []
    for i in range(len(c)):
        subsets.append([x for i_,x in enumerate(c) if i_ != i])
    return subsets
        
def MScandidate_gen(Fk_less1, MS, sdc, not_together, must_have):
    C = []
    n = len(Fk_less1)
    nk = len(Fk_less1[0])    # Get size of k-1

    for i in range(n):
        for j in range(n):
            if i == j: continue

            f1 = Fk_less1[i]    # TODO: verify if the swap needs to be considered as well.
            f2 = Fk_less1[j]

            if f1[0:nk - 1] == f2[0:nk - 1] and f1[-1] < f2[-1] and \
                sdcSatisfied(supportCount([f1[-1]], T), supportCount([f2[-1]], T), T, sdc):
                c = [x for x in f1]     # Deep copy
                c.append(f2[-1])        # f1 U f2

                # Check pruning condition before appending
                should_append = True
                for s in kLess1Subsets(c):
                    if (c[0] in s or MS[c[0]] == MS[c[1]]) and s not in Fk_less1:
                        should_append = False

                # Check constraints
                if should_append:
                    C.append(c)
    return C

import os
result_file = "result.txt";

try:
    os.remove(result_file)
except OSError:
    pass

file = open(result_file, "w") 
 
def printFrequentkItemsets(Fk, k, item_counts, tail_counts, not_together, must_have):
    file.write("\nFrequent %d-itemsets\n" % k)
    counter = 0

    for i, itemset in enumerate(Fk):
        if not notTogetherSatisfied(not_together, itemset) or not mustHaveSatisfied(must_have, itemset):
            continue
        file.write("%d : %s\n" % (item_counts[i], itemset))
        if len(tail_counts) > 0:
            file.write("Tailcount = %d\n\n" % tail_counts[i])
        counter += 1

    file.write("\nTotal no. of frequent-%d itemsets = %d\n" % (k, counter))

def MSapriori(T, MS, sdc, not_together, must_have):
    temp_s = set()
    for t in T: 
        for x in t: temp_s.add(x)
    
    M = sorted(list(temp_s), cmp=lambda x,y: -1 if MS[x] - MS[y] < 0 else 1)     # Uses Python < 3.x
    L, L_counts = initPass(M, T, MS)
    Fk = []   # F_k-1
    Fk1_counts = []
    F = []      # For union over all Fk

    # Populate F1
    for i,x in enumerate(L):
        if support(L_counts[i], T) >= MS[x]:
            Fk.append([x])
            Fk1_counts.append(L_counts[i])

    printFrequentkItemsets(Fk, 1, Fk1_counts, [], not_together, must_have)

    k = 2
    while len(Fk) > 0:
        C = level2_candidate_gen(L, L_counts, T, MS, sdc, not_together, must_have) if k == 2 else \
            MScandidate_gen(Fk, MS, sdc, not_together, must_have)
        c_counts = []
        tail_counts = []

        for c in C:
            c_counts.append(supportCount(c, T))
            tail_counts.append(supportCount(c[1:], T))  # Persist this in global array if rule gen reqd.

        Fk = []
        c_counts_temp = []
        t_counts_temp = []

        for i,c in enumerate(C):
            if support(c_counts[i], T) >= MS[c[0]]:
                Fk.append(c)
                c_counts_temp.append(c_counts[i])
                t_counts_temp.append(tail_counts[i])
        printFrequentkItemsets(Fk, k, c_counts_temp, t_counts_temp, not_together, must_have)
        [F.append(x) for x in Fk if x not in F]
        k += 1


#T = [[20, 30, 80, 70, 50, 90],
#    [20, 10, 80, 70],
#    [10, 20, 80],
#    [20, 30, 80],
#    [20, 80],
#    [20, 30, 80, 70, 50, 90, 100, 120, 140]]

#MIS = {}
#MIS[10] = 0.43
#MIS[20] = 0.30
#MIS[30] = 0.30
#MIS[40] = 0.40
#MIS[50] = 0.40
#MIS[60] = 0.30
#MIS[70] = 0.20
#MIS[80] = 0.20
#MIS[90] = 0.20
#MIS[100] = 0.10
#MIS[120] = 0.20
#MIS[140] = 0.15
#SDC = 0.1
#cannot_be_together = [[20, 40], [70, 80]]
#must_have = [20, 40, 50, 70, 140]
from reader import Reader
T = Reader.InputData
MIS, SDC, cannot_be_together, must_have = Reader.ParameterData
MSapriori(T, MIS, SDC, cannot_be_together, must_have)
file.close()