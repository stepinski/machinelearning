
def arrayManipulation(n, queries):
    # Write your code here
    arr = [0]*n
    for q in queries:
         a=q[0]-1
         b=q[1]
         leng=b-a
         tmp=[q[2]]*leng
         arr[q[0]-1:q[1]]+=tmp
    return max(arr)
        
