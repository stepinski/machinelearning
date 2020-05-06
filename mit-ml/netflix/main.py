import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")

# TODO: Your code here
# for K in [1,2,3,4]:
#     for seed in [0,1,2,3,4]:
#         mixture,post=common.init(X, K, seed)   
#         mixture, post, cost=kmeans.run(X,mixture,post)
#         common.plot(X,mixture,post,title='K=%s seed=%s cost=%s'%(K,seed,cost))
#         print('K=%s seed=%s cost=%s'%(K,seed,cost))

for K in [1,2,3,4]:
    maxcost=-100000
    for seed in [0]:
        mixture,post=common.init(X, K, seed)   
        mixture, post, cost=naive_em.run(X,mixture,post)
        print(common.bic(X,mixture,cost))
        # common.plot(X,mixture,post,title='EM K=%s seed=%s cost=%s'%(K,seed,cost))
        # mixture,post=common.init(X, K, seed)   
        # mixture, post, cost=kmeans.run(X,mixture,post)
        # common.plot(X,mixture,post,title='kmeansK=%s seed=%s cost=%s'%(K,seed,cost))
        # print('K=%s seed=%s cost=%s'%(K,seed,cost))
        # if cost>maxcost: maxcost=cost
    # print("maxll %s"%maxcost)
# K=3
# seed=0
# mixture,post=common.init(X, K, seed)  
# # print(mixture)
# # posts,ll = naive_em.estep(X,mixture)
# # print('ll = %s'%ll)
# # print(posts)
# # for K in [1,2,3,4]:
# #     for seed in [0,1,2,3,4]:
# #         mixture,post=common.init(X, K, seed)   
#         # mixture, post, cost=kmeans.run(X,mixture,post)
#         # #common.plot(X,mixture,post,title='K=%s seed=%s cost=%s'%(K,seed,cost))
#         # print('K=%s seed=%s cost=%s'%(K,seed,cost))

# K=5
# X = np.loadtxt("lasttestx.txt")
# mu=np.array( [[-0.60787456, 0.09534884],
#  [ 0.53830805, -0.24498689],
#  [ 0.4983494,  -0.94992061],
#  [-0.66868763, -0.9861811 ],
#  [-0.15367443, -0.44492439]])

# var=np.array([0.66695384, 0.30533997, 1.00062913, 1.639639,0.61075705])
# p=np.array([0.12075413,0.26092829, 0.19481629, 0.23742157, 0.18607972])

# mixture = common.GaussianMixture(mu, var, p)
# posts,ll = naive_em.estep(X,mixture)
# print('ll = %s'%ll)
# print(posts)

# newmixt=naive_em.mstep(X,posts)
# print(newmixt)


 
# n=X.shape[0]
# llt=0.0
# # print("startar")
# # for i in range(n):
# #     for j in range(K):
# #         llt+=np.log(mixture.p[j]*Gaussian(mixture[i,j], var[j],X[i])
# #     # print(pn)
# #     # print(pn.sum())
# #     llt+=np.log(pn)
# # print("test %.20f"%llt)
# # print(llt)

# # print('fin')

# # # Output:
# # # post:[[0.03939317 0.66938479 0.1207385  0.05606209 0.11442145]
# # #  [0.1284887  0.46274858 0.09891089 0.08438805 0.22546379]
# # #  [0.12250705 0.49162696 0.09739513 0.0799134  0.20855745]
# # #  [0.0496701  0.65425613 0.1051122  0.05541291 0.13554867]
# # #  [0.09493723 0.56463229 0.10373629 0.07240648 0.16428772]
# # #  [0.17238229 0.41053463 0.10757978 0.10210253 0.20740077]
# # #  [0.20502453 0.35053858 0.10083158 0.10866167 0.23494364]
# # #  [0.04599863 0.66358567 0.10870553 0.05489602 0.12681415]
# # #  [0.11717788 0.40929401 0.18426962 0.13553269 0.15372579]
# # #  [0.19227515 0.37045863 0.11410566 0.11445474 0.20870582]
# # #  [0.13920751 0.47399095 0.10541633 0.08908133 0.19230388]]
# # # LL:-25.086211


# # tst=np.array([[0.03939317, 0.66938479, 0.1207385,  0.05606209, 0.11442145],
# #  [0.1284887,  0.46274858, 0.09891089, 0.08438805, 0.22546379],
# #  [0.12250705, 0.49162696, 0.09739513, 0.0799134 , 0.20855745],
# #  [0.0496701,  0.65425613, 0.1051122 , 0.05541291, 0.13554867],
# #  [0.09493723, 0.56463229, 0.10373629, 0.07240648, 0.16428772],
# #  [0.17238229, 0.41053463, 0.10757978, 0.10210253, 0.20740077],
# #  [0.20502453, 0.35053858, 0.10083158, 0.10866167, 0.23494364],
# #  [0.04599863, 0.66358567, 0.10870553, 0.05489602, 0.12681415],
# #  [0.11717788, 0.40929401, 0.18426962, 0.13553269, 0.15372579],
# #  [0.19227515, 0.37045863, 0.11410566, 0.11445474, 0.20870582],
# #  [0.13920751, 0.47399095, 0.10541633, 0.08908133, 0.19230388]])
 
# # n=tst.shape[0]
# # print(n)
# # llt=0.0
# # print("bla")
# # for i in range(n):
# #     print(tst[i,:])
# #     print(np.log(tst[i,:]).sum())
# #     print("end")
# #     pn=np.log(tst[i,:]).sum()
# #     # print(pn)
# #     # print(pn.sum())
# #     llt+=pn
# # print("test %.20f"%llt)
# # print(llt)