n, m = map(int, input().split())
A = []
for i in range(n):
    a = list(map(int, input().split()))
    A.append(a)
for i in range(m):
    for j in range(n - 1, -1, -1):
        print(A[j][i],end=' ')
    print()