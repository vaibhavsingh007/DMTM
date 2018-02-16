
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

support = lambda count, T: float(count)/len(T)

def initPass(M, T, MS):
    l = []
    l_count = []    # to memoize the support count of l

    for i,m in enumerate(M):
        sup_count = supportCount([m], T)

        if support(support_count, T) >= MS[m]:
            # Add m to l, and all subsequent j"s if sup(j) >= MS[m]
            l.append(m)
            l_count.append(sup_count)

            for j in M[i+1:]:
                sup_count = supportCount([j], T)

                if support(sup_count, T) >= MS[m]:
                    l.append(j)
                    l_count.append(sup_count)
            break
    return l, l_count

def level2_candidate_gen(L, L_counts, T, MS):
    C = []

    for i,l in enumerate(L):
        if support(L_counts[i], T) >= MS[l]:
            for i_,j in enumerate(L[i+1:]):
                if support(L_counts[i_], T) >= MS[l]:
                    C.append([l,j])
    return C


def MSapriori(T, MS):
    temp_s = {}
    [temp_s.add(i) for i in [t for t in T]]     # T should be a 2D array
    
    M = sorted(list(temp_s), cmp=lambda x,y: MS[x] - MS[y])     # Uses Python < 3.x
    L, L_counts = initPass(M, T, MS)
    Fk = [x for i,x in enumerate(L) if float(L_counts[i])/len(T) >= MS[x]]   # F_k-1
    F = {}      # union over all Fk

    k = 2
    while len(F) > 0:
        C = level2_candidate_gen(L, L_counts) if k == 2 else MScandidate_gen(Fk)
        C_counts = []
        tail_counts = []

        for c in C:
            C_counts.append(supportCount(c, T))
            tail_counts.append(supportCount(c[1:], T))  # Persist this in global array if rule gen reqd.

        Fk = []
        [Fk.append(c) for i,c in enumerate(C) if float(C_counts[i])/len(T) > MS[c[0]]]
        [F.add(i) for i in Fk]


T = [[20, 30, 80, 70, 50, 90],
    [20, 10, 80, 70],
    [10, 20, 80],
    [20, 30, 80],
    [20, 80],
    [20, 30, 80, 70, 50, 90, 100, 120, 140]]

MIS = {}
MIS[10] = 0.43
MIS[20] = 0.30
MIS[30] = 0.30
MIS[40] = 0.40
MIS[50] = 0.40
MIS[60] = 0.30
MIS[70] = 0.20
MIS[80] = 0.20
MIS[90] = 0.20
MIS[100] = 0.10
MIS[120] = 0.20
MIS[140] = 0.15
SDC = 0.1
cannot_be_together = [[20, 40], [70, 80]]
must_have = [20, 40, 50]
