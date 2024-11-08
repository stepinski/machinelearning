from collections import Counter

def makeAnagram(a, b):
    # Count the frequency of characters in both strings
    count_a = Counter(a)
    count_b = Counter(b)
    
    # Calculate the total number of deletions
    deletions = 0
    
    # Calculate the differences in frequencies for each character
    for char in set(a + b):  # Use set to avoid checking the same character twice
        deletions += abs(count_a[char] - count_b[char])
    
    return deletions

