# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import random
test = 232
print(test*10)

# generate all combinations of N items
def powerSet(items):
    N = len(items)
    # enumerate the 2**N possible combinations
    for i in range(2**N):
        combo = []
        for j in range(N):
            # test bit jth of integer i
            if (i >> j) % 2 == 1:
                combo.append(items[j])
        yield combo


def powerSet2(items):
    N = len(items)
    # enumerate the 2**N possible combinations
    for i in range(3**N):
        combo = []
        combo2 = []
        for j in range(N):
            # test bit jth of integer i
            if ( i // 3**j ) % 3 == 1:
                combo.append(items[j])
            elif ( i // 3**j ) % 3 == 2:
                combo2.append(items[j])
        yield combo,combo2
        
        
def yieldAllCombos(items):
    #split items and generate powersets for setA and setB
    def pset(items):
        N = len(items)
        # enumerate the 2**N possible combinations
        for i in range(3**N):
            combo = []
            combo2 = []
            for j in range(N):
            # test bit jth of integer i
                if ( i // 3**j ) % 3 == 1:
                    combo.append(items[j])
                elif ( i // 3**j ) % 3 == 2:
                    combo2.append(items[j])
            yield combo,combo2
    cmbs = pset(items)
    for cmb in cmbs:
        listA=cmb
        listB=list(set(items)-set(cmb))
        cmB=pset(listB)
        for c in cmB:
            yield listA,c

#the correct one!!!
def yieldAllCombos2(items):
    #split items and generate powersets for setA and setB
    N = len(items)
    # enumerate the 2**N possible combinations
    for i in range(3**N):
        combo = []
        combo2 = []
        for j in range(N):
            # test bit jth of integer i
            if ( i // 3**j ) % 3 == 1:
                combo.append(items[j])
            elif ( i // 3**j ) % 3 == 2:
                combo2.append(items[j])
        yield combo,combo2
       
class Item(object):
    def __init__(self, n, v, w):
        self.name = n
        self.value = float(v)
        self.weight = float(w)
    def getName(self):
        return self.name
    def getValue(self):
        return self.value
    def getWeight(self):
        return self.weight
    def __str__(self):
        return '<' + self.name + ', ' + str(self.value) + ', '\
                     + str(self.weight) + '>'
          

def buildRandomItems(n):
    return [Item(str(i),10*random.randint(1,10),random.randint(1,10))
            for i in range(n)]
    
def buildItems():
    return [Item(n,v,w) for n,v,w in (('clock', 175, 10),
                                      ('painting', 90, 9),
                                      ('radio', 20, 4),
                                      ('vase', 50, 2),
                                      ('book', 10, 1),
                                      ('computer', 200, 20))]
def buildItems2():
    return [Item(n,v,w) for n,v,w in (('clock', 175, 10),
                                      ('painting', 90, 9),
                                      ('radio', 20, 4)
                                     )]
    
def printItem(title,ps):
    test=""
    for p in ps:
        test = test+ ' {0}'.format(str(p))
    print title + test

def printps(ps):
    for a,b in ps:
       printItem("contenta:",a)
       printItem("contentb:",b)