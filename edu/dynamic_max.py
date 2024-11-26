def maxSubsetSum(arr):
    if not arr: return 0
    if len(arr)==1: return max(0,arr[0])
    prev2 = max(0,arr[0])
    prev1=max(prev2,arr[1])
    for i in range(2,len(arr)):
        current = max(prev1,arr[i]+prev2)
        prev2=prev1
        prev1=current
    return prev1
