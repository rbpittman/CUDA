#!/bin/bash
# 1) PyCPU
# 2) C++CPU
# 3) PyCUDA
# 4) CUDA

SELECTION=2

TRIALS=10
IMAGE_START=100
IMAGE_END=100
IMAGE_STEP=50
IMAGE_DEPTH=500

FILENAME="notSet"

#Pre: One argument must be specified to this function 
#     containing the program to run, e.g. "python mand.py".
#Post: Runs the program with the global args specified at the
#      top of this file, and outputs status as well as 
#      relevant benchmark data to $FILENAME
printStdDev() {
    echo "Computing stddev, storing output in \"$FILENAME\".";
    echo "Trials per format: $TRIALS" >> $FILENAME;
    local CURR_IMAGE=$IMAGE_START
    local ALL_TIME_DATA=""
    while [ $CURR_IMAGE -le $IMAGE_END ]; do
	echo "${CURR_IMAGE} X ${CURR_IMAGE}, depth=$IMAGE_DEPTH" >> $FILENAME;
	echo "${CURR_IMAGE} X ${CURR_IMAGE}, depth=$IMAGE_DEPTH";
        #Do the individual trials:
	for trial in `seq $TRIALS`; do
	    echo -n "Trial $trial: ";
	    local PARAMS="$CURR_IMAGE $CURR_IMAGE $IMAGE_DEPTH";
	    #COMMAND="$TIME_COMMAND $PROG $PARAMS";
	    local COMMAND="$1 $PARAMS";
	    echo "Command to run: $COMMAND";
	    local RESULT=$((time $COMMAND) 2>&1);
	    local GREP_FLOATS=`echo $RESULT | grep -o "[0-9]*m[0-9]*\.[0-9]*"`
	    local TIME=`echo $GREP_FLOATS | grep -o "^[0-9]*m[0-9]*\.[0-9]*"`
	    #Just store the times in the file
	    echo -n "$TIME" >> $FILENAME;
	    echo $TIME;
	    if [ $trial = $TRIALS ]; then
		echo "" >> $FILENAME;
	    else
		echo -n " " >> $FILENAME;
	    fi
	done
	let CURR_IMAGE+=IMAGE_STEP
    done
}

#ASSERT: N = 1,2,3, or 4
#SETUP FILE HEADER:
if [ $SELECTION = 1 ]; then #PyCPU
    FILENAME="stddev_PyCPU.txt"
    echo "PyCPU STDDEV" > $FILENAME;
    printStdDev "python mand.py"
elif [ $SELECTION = 2 ]; then #C++CPU
    FILENAME="stddev_C++CPU.txt"
    echo "C++CPU STDDEV" > $FILENAME;
    printStdDev "./mand"
elif [ $SELECTION = 3 ]; then #PyCUDA
    FILENAME="stddev_PyCUDA.txt"
    echo "PyCUDA STDDEV" > $FILENAME;
    printStdDev "python gpuMand.py"
elif [ $SELECTION = 4 ]; then #CUDA
    FILENAME="stddev_CUDA.txt"
    echo "CUDA STDDEV" > $FILENAME;
    printStdDev "./gpuMand"
fi

exit 0;
