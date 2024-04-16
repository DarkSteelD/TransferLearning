def find_max_min_lg2(b):
     #print(b)
     if len(b) == 2 or len(b) // 2 <= 1:
         return b.index(min(b)), b.index(max(b))
     
     min1,max1 = find_max_min_lg2(b[0:len(b)//2])
     min2,max2 = find_max_min_lg2(b[len(b)//2:len(b)])
     min2 += len(b)//2
     max2 += len(b)//2
     mina = min2 if b[min2] < b[min1] else min1
     maxa = max2 if b[max2] > b[max1] else max1
     return mina,maxa
def ans():
    n,q = map(int,input().split())
    a = list(map(int,input().split()))
    for _ in range(q):
        r,l = map(int,input().split())
        k,l =  find_max_min_lg2(a[r-1:l])
        k+=r-1
        l+=r-1
        #print(k,l)
        a[k], a[l] = a[l], a[k]
    print(' '.join(map(str,a)))
ans()