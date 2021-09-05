out = []
nums = [1,2,3]

def dfs(a,temp):
    if a == []:
        out.append(temp)
    else:
        for i in a:
            temp.append(i)
            temp2 = temp
            a.remove(i)
            a2 = a
            dfs(a2,temp2)

dfs(nums,[])
print(out)