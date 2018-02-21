import sys

def extractInputData(filename):
    f = open(filename, 'r')
    T = []

    for l in f.readlines():
        digits = l.replace('\n','').replace('{','').replace('}','').split(',')
        T.append([int(d) for d in digits])
    return T

def extractParameterData(filename):
    f = open(filename, 'r')
    MS = {}
    sdc = None
    not_together = []
    must_have = None

    for l in f.readlines():
        if l.startswith('MIS'):
            splits = l.split()
            s = splits[0]
            item = int(s[s.find("(")+1:s.find(")")])
            mis = float(splits[2])
            MS[item] = mis

        if l.startswith('SDC'):
            sdc = float(l.split()[2])

        if l.startswith('cannot_be_together'):
            set_splits = [x.replace('{','').replace('}','').replace(',','').replace('\n','').strip() for x in l[l.find(":")+1:].split('},')]        #['20 40', '70 80']

            for s in set_splits:
                not_together.append([int(x) for x in s.split()])

        if l.startswith('must-have'):
            splits = l[l.find(":")+1:].split('or')
            must_have = [int(x.strip()) for x in splits]

    return MS, sdc, not_together, must_have

class Reader:
    InputData = extractInputData(r'C:\Users\vaibh\Downloads\MsApriori Data\3\input-data.txt')
    ParameterData = extractParameterData(r'C:\Users\vaibh\Downloads\MsApriori Data\3\parameter-file.txt')

    #C:\Users\vaibh\Downloads\MsApriori Data\2\


