T = int(input())
res = []
for i in range(T):
    I = input().split(" ")
    n, m, C = int(I[0]), int(I[1]), int(I[2])
    a = []
    for i in range(n):
        a.append(input().split(" "))
    M = 0
    for w in range(1, m):
        h = int(C/2 - w)
        for x in range(0, m-w):
            for y in range(0, n-h):
                s=0
                
                for i in a[y:y+h]:
                    for j in i[x:x+w]:
                        s += int(j)
                if s > M:
                    M = s
    res.append(M)
for i in res:
    print(i)
    
    