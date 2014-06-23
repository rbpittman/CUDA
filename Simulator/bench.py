import os

OUTPUT = "sim_bench.txt"

with open(OUTPUT, "w") as out:
    for num_particles in range(10, 500, 25):
        print "Running with " + str(num_particles) + " particles"
        os.system("python randomParticles.py " + str(num_particles))
        simTime = os.popen("./simulator").read()
        cpuTime = os.popen("./cpuSimulator").read()
        print "gpu: " + str(simTime) + ", cpu: " + str(cpuTime)
        out.write(simTime + ' ' + cpuTime + '\n')
