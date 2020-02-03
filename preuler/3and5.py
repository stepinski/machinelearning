import functools
a = [i for i in range(1000)]
fss = functools.reduce(lambda a,b: a+b, map(lambda x:  0 if x%5 >0 else x, a))
ts = map(lambda x:  0 if x%3 >0 else x, a)
tss = functools.reduce(lambda a,b: a+b, map(lambda x:  0 if x%3 >0 else x, a))
com= map(lambda x:  0 if x%(3*5) >0 else x, a)
coms = functools.reduce(lambda a,b: a+b, map(lambda x:  0 if x%(3*5) >0 else x, a))
print (fss+tss-coms)
