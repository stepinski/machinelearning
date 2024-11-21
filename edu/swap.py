def countSwaps(a):
    # Write your code here
    n=len(a)
    cnt_swaps=0
    
    for i in range(0, n):
    
        for j in range(0, n - 1):
            if a[j] > a[j + 1]:
                a[j],a[j+1]=a[j+1],a[j]
                cnt_swaps+=1

    print(f"Array is sorted in {cnt_swaps} swaps.")
    print(f"First Element: {a[0]}")
    print(f"Last Element: {a[-1]}")
