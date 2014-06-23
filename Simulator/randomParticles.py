import random
import sys

#n = 512
n = int(sys.argv[1])
low = 200
high = 400

f = open("particles.txt", "w")
f.write(str(n) + '\n')
for i in range(n):
    f.write(str(random.randint(low, high)) + ' ' + str(random.randint(low, high)) + '\n')

f.close()
