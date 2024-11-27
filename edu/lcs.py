def longestCommonSubsequence(A, B):
    n = len(A)
    m = len(B)
    
    # Initialize DP table
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    
    # Fill the DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if A[i - 1] == B[j - 1]:  # Matching elements
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:  # Not matching
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    # Reconstruct the LCS
    lcs = []
    i, j = n, m
    while i > 0 and j > 0:
        if A[i - 1] == B[j - 1]:
            lcs.append(A[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    
    return list(reversed(lcs))

# Example usage:
A = [1, 2, 3, 4, 1]
B = [3, 4, 1, 2, 1, 3]
print(longestCommonSubsequence(A, B))  # Output: [1, 2, 3]

