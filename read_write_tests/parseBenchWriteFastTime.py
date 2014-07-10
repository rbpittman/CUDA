f = open("benchWriteFastTime.txt", 'r')
g = open("benchGlobalWriteFastTime.txt", 'w')
reg = open("benchRegWriteFastTime.txt", 'w')

g.write("benchmark efficient global\n")
g.write("Trials: 3\n")
reg.write("benchmark efficient register\n")
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
