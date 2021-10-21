fo = open("out/result.txt", "a")

for i in range(10):
    fo.write(str(i)+'\n')

fo.close()