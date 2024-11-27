def weightedMean(X, W):
    # Write your code here
    wmean= sum(x*w for x, w in zip(X,W))/sum(W)
    print(f"{wmean:.1f}")

