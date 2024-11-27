# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import Counter 
# Read input
n = int(input().strip())  # Number of elements
arr = list(map(int, input().strip().split()))  # List of integers
# Calculate Mean
mean = sum(arr) / n

# Calculate Median
arr.sort()
if n % 2 == 0:
    median = (arr[n // 2 - 1] + arr[n // 2]) / 2
else:
    median = arr[n // 2]

# Calculate Mode
frequency = Counter(arr)
max_freq = max(frequency.values())
mode_candidates = [key for key, value in frequency.items() if value == max_freq]
mode = min(mode_candidates)

# Print results
print(f"{mean:.1f}")
print(f"{median:.1f}")
print(mode)
