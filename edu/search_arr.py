def balancedSums(arr):
    total_sum = sum(arr)  # Calculate the total sum of the array
    left_sum = 0  # Initialize left sum to 0

    for i in range(len(arr)):
        # Right sum can be derived as:
        right_sum = total_sum - left_sum - arr[i]

        if left_sum == right_sum:
            return "YES"
        
        # Update left_sum for the next iteration
        left_sum += arr[i]

    return "NO"

