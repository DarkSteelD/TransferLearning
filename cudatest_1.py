N = int(input())
l = []
for _ in range(N):
    a,b = map(float, input().split())
    l.append((a,b))
l.sort(key=lambda x: (x[0],x[1]))
print(l)
rg = 0
coun = 0
for i in l:
    if rg < i[0]:
        rg = i[1]
        coun+=1
    if rg > i[1]:
        rg = i[1]
print(coun)