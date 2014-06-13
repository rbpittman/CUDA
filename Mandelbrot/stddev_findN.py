#This program goes through potential values of n, starting from n=5, 
#then runs the current stddev_get_times.sh program, collects the data,
#then finds the confidence interval for that data. If less than 1% of 
#sample average, stops and prints the interval and prints n. Otherwise
#n++ and repeat. 

#11, 20, 19, 20, 24, 22, 17, 18, 
#[13, 10, 12, 14, 25, 13, 12, 17, 19, 18]

import os

def average(nums):
    return sum(nums) / len(nums)

def variance(nums):
    variances = map(lambda x: (x - average(nums)) ** 2, nums)
    return average(variances)

NUM_TRIALS_OF_GET_N = 10
FILE = "stddev_test.txt"
n_results = []

for i in range(NUM_TRIALS_OF_GET_N):
    percentConf = 1.
    n = 4
    while percentConf > 0.01:
        n += 1
        print "Trying n ==", n
        command = "./bench.sh -f -o " + FILE + " -s 2 -n " + str(n) + " > /dev/null"
        print "  command ->", command
        os.system(command)
        f = open(FILE, 'r')
        f.readline()#Get rid of bench type
        f.readline()#Get rid of num trials, we already know
        data_set_num = 0
        for line in f:
            time_list = line.strip().split(' ')
            if time_list[1] != 'X':
                times = []
                for string in time_list:
                    mins = int(string[:string.index('m')])
                    secs = float(string[string.index('m') + 1:])
                    times.append((60 * mins) + secs)
                var = variance(times)
                mean = average(times)
                radius = (1.96 * (var**0.5)) / (n ** 0.5)
                percentConf = radius / mean
                print "  found radius", radius, "with percentage", str(100 * percentConf) + "%"
    print n
    n_results.append(n)
print n_results
