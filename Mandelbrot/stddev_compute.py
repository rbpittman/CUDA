import sys

#NUM_SIZES = 5 #The number of sets of trials

def average(nums):
    return sum(nums) / len(nums)

def variance(nums):
    variances = map(lambda x: (x - average(nums)) ** 2, nums)
    return average(variances)

if len(sys.argv) != 2:
    print "Arg error"
else:
    filename = sys.argv[-1]
    f = open(filename, 'r')
    f.readline()
    num_trials = int(f.readline().strip().split(" ")[-1])
    data_set_num = 0
    for line in f:#for i in range(NUM_SIZES):
        time_list = line.strip().split(' ')#f.readline().strip().split(' ')
        if time_list[1] != 'X':
            data_set_num += 1
            times = []
            for string in time_list:
                mins = int(string[:string.index('m')])
                secs = float(string[string.index('m') + 1:])
                times.append((60 * mins) + secs)
            print "======================"
            # print "first to second diff:", times[1]-times[0]
            #times = times[len(times)//2:]
            print "Data set", data_set_num, "->", times
            var = variance(times)
            mean = average(times)
            print "variance:", var
            print "avg:", mean
            #===================NEW FORMULA
            radius = (1.96 * (var**0.5)) / (num_trials ** 0.5)
            print "RADIUS:", radius
            #===================
            
            # E = 0.01 * mean # Don't let deviate by more than 1% 
            # n = var / (0.05 * (E**2))
            # n = var / (0.05 * ((0.01 * mean)**2))
            # E = (var / (0.05 * n)) ** 0.5
            print "95% chance that our average is within", round(radius, 5), "seconds of actual average"
            print "95% chance of error within", str(round(100 * (radius / mean), 4)) + "%"
            # print "n:", n
            # print "diffs:",
            # for i in range(1, len(times)):
            #     print times[i] - times[i - 1],
            print
    f.close()
