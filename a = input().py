n = int(input())
id,mass = [],[]
leng = 0
id2 = []
for _ in range(n):
    id.append(int(input()))
    c = list(map(int, input().split()))
    mass.append(c)
    leng += id[_]
    id2.append(c[0])
k = []
for i in range(leng):
    mv = 100001
    ll = 0
    for i in id2:
        mv = min(i,mv)
        ll = i
    k.append(mv)
    id2[ll] = mass
print(' '.join(str(i) for i in k))