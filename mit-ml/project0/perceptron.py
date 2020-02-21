import numpy as np


def percept(x, y, theta):
    progress = []
    mistakes = 0
    iter = 0
    # progress.append(theta)
    stopit = False
    cnt=0
    while(not stopit):
        iter=0
        for i in x:
            if np.dot(np.transpose(theta), i)*y[iter] <= 0:
                theta = theta+y[iter]*i
                progress.append(theta)
                mistakes += 1
                cnt=0
            else: cnt+=1
            
            if(cnt==len(x)): 
                stopit=True
                break
            iter += 1

    return (mistakes, progress)

def test1():
    x = np.array([[-1,-1],[1, 0], [-1, 1.5]])
    y = [1, -1, 1]
    theta = np.array([0,0])
    res = percept(x, y, theta)
    print("test1:")
    print(res[0])
    print(res[1])

def test2():
    x = np.array([[1, 0], [-1, 1.5],[-1,-1]])
    y = [-1, 1, 1]
    theta = np.array([0,0])
    res = percept(x, y, theta)
    print("test2:")
    print(res[0])
    print(res[1])

def test3():
    x = np.array([[-1,-1],[1, 0], [-1, 10]])
    y = [1, -1, 1]
    theta = np.array([0,0])
    res = percept(x, y, theta)
    print("test3:")
    print(res[0])
    print(res[1])

def test4():
    x = np.array([[1, 0], [-1, 10],[-1,-1]])
    y = [-1, 1,1]
    theta = np.array([0,0])
    res = percept(x, y, theta)
    print("test4:")
    print(res[0])
    print(res[1])

def main():
    test1()
    test2()
    test3()
    test4()


if __name__ == "__main__":
    main()
