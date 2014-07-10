f = open("benchReadFastTime.txt", 'r')
g = open("benchGlobalReadFastTime.txt", 'w')
reg = open("benchRegReadFastTime.txt", 'w')

g.write("benchmark efficient global read\n")
g.write("Trials: 3\n")
reg.write("benchmark efficient register read\n")
reg.write("Trials: 3\n")

f.readline()
lines = f.readlines()
x = 0
for line in lines:
    if line.strip().isdigit():
        g.write(line)
        reg.write(line)
    else:
        line = line.strip().split(' ')
        g_times = line[5::4]
        for i, time in enumerate(g_times):
            g.write("0m" + time)
            if i != len(g_times) - 1:
                g.write(' ')
            else:
                g.write('\n')
        reg_times = line[7::4]
        for i, time in enumerate(reg_times):
            reg.write("0m" + time)
            if i != len(reg_times) - 1:
                reg.write(' ')
            else:
                reg.write('\n')
f.close()
g.close()
reg.close()
