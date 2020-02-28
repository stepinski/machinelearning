import numpy as np


def percept(x, y, theta):
    progress = []
    mistakes = 0
    iter = 0
    # progress.append(theta)
    stopit = False
    cnt = 0
    while(not stopit):
        iter = 0
        for i in x:
            if np.dot(np.transpose(theta), i)*y[iter] <= 0:
                theta = theta+y[iter]*i
                progress.append(theta)
                mistakes += 1
                cnt = 0
            else:
                cnt += 1

            if(cnt == len(x)):
                stopit = True
                break
            iter += 1

    return (mistakes, progress)


def percept2(x, y, theta, th0):
    progress = []
    mistakes = {}
    iter = 0
    # progress.append(theta)
    stopit = False
    cnt = 0
    while(not stopit):
        iter = 0
        for i in x:
            if (np.dot(np.transpose(theta), i)+th0)*y[iter] <= 0:
                theta = theta+y[iter]*i
                th0 = th0+y[iter]
                progress.append((theta, th0))
                key = str(i)
                mistakes[key] = 1 if key not in mistakes else mistakes[key] + 1
                # mistakes += 1
                cnt = 0
            else:
                cnt += 1
            print(mistakes)
            print(str(theta) + "-" + str(th0))

            if(cnt == len(x)):
                stopit = True
                break
            iter += 1

    return (mistakes, progress)

def perceptlast(x, y, theta):
    progress = []
    mistakes = 0
    iter = 0
    # progress.append(theta)
    stopit = False
    cnt = 0
    while(not stopit):
        iter = 0
        for i in x:
            if np.dot(np.transpose(theta), i)*y[iter] <= 0:
                theta = theta+y[iter]*i
                progress.append(theta)
                mistakes += 1
                cnt = 0
            else:
                cnt += 1

            if(cnt == len(x)):
                stopit = True
                break
            iter += 1

    return (mistakes, progress)


def testalfa():
    x = np.array([[np.cos(np.pi), 0], [0, np.cos(2*np.pi)]])
    y = [1, 1]
    theta = np.array([0, 0])
    res = percept(x, y, theta)
    print("test1:")
    print(res[0])
    print(res[1])


def test1():
    x = np.array([[-1, -1], [1, 0], [-1, 1.5]])
    y = [1, -1, 1]
    theta = np.array([0, 0])
    res = percept(x, y, theta)
    print("test1:")
    print(res[0])
    print(res[1])


def test2():
    x = np.array([[1, 0], [-1, 1.5], [-1, -1]])
    y = [-1, 1, 1]
    theta = np.array([0, 0])
    res = percept(x, y, theta)
    print("test2:")
    print(res[0])
    print(res[1])


def test3():
    x = np.array([[-1, -1], [1, 0], [-1, 10]])
    y = [1, -1, 1]
    theta = np.array([0, 0])
    res = percept(x, y, theta)
    print("test3:")
    print(res[0])
    print(res[1])


def test4():
    x = np.array([[1, 0], [-1, 10], [-1, -1]])
    y = [-1, 1, 1]
    theta = np.array([0, 0])
    res = percept(x, y, theta)
    print("test4:")
    print(res[0])
    print(res[1])


def testt0():
    x = np.array([[-4, 2], [-2, 1], [-1, -1], [2, 2], [1, -2]])
    y = [1, 1, -1, -1, -1]
    theta = np.array([0, 0])
    th0 = 0
    res = percept2(x, y, theta, th0)
    print("testt0:")
    print(res[0])
    print(res[1])


def testnoerror():
    x = np.array([[-4, 2], [-2, 1], [-1, -1], [2, 2], [1, -2]])
    y = [1, 1, -1, -1, -1]
    theta = np.array([-3, 3])
    th0 = -3
    res = percept2(x, y, theta, th0)
    print("testerror:")
    print(res[0])
    print(res[1])


def main():
    # test1()
    # test2()
    # test3()
    # test4()
    # testt0()
    # testnoerror()
    testalfa()


if __name__ == "__main__":
    main()
