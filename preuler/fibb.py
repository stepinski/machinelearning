import functools
import operator
foldl = lambda func,acc,xs:functools.reduce(func,xs,acc)
a = [i for i in range(4000000)]
print(foldl(operator.add,0,a))
