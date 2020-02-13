def calcbalance(start,mperc,interest,paid):
    balance=start
    for i in range(12):
        #paid=balance*mperc
        remaining=balance-paid
        balance=remaining*interest/12.0+remaining
    #print("Remaining balance: %.2f" % balance)
    return balance
def bsearch(bal,ai):
    n=1
    nmax=100000000000
    start=((bal/12.0))
    end=((bal+bal*ai)/12.0)+1
    tol=0.0001
    ret=1000000
    while(n<nmax):
        c=(end+start)/2.0
        # print(c)
        tmp = calcbalance(bal,2,ai,c)
        if(tmp==0 or (end-start)/2<tol) : 
            # print(tmp)
            # print((end-start)/2<tol)
            ret=c
            break
        n += 1
        tmp2= calcbalance(bal,2,ai,start)
        if (tmp>0 and tmp2>0) : start=c 
        else : end=c

    return ret 
def main():

    balance = 320000; annualInterestRate = 0.2; monthlyPaymentRate = 0.04
    bal=100
    end=round((balance+balance*annualInterestRate)/12)+10
    ret=100
    
    ret =bsearch(balance,annualInterestRate)
    
    # for p in range(0,end,10):
    #     bal = calcbalance(balance,monthlyPaymentRate,annualInterestRate,p)
    #     if(bal<0):
    #         ret=p
    #         break
    #calcbalance(balance,monthlyPaymentRate,annualInterestRate)
    print("Lowest Payment: %.2f" % ret)
if __name__ == "__main__":
    main()