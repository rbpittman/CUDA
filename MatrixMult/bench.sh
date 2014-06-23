#!/bin/bash
#-f option forces overwrite of whichever filename is 
#    specified (if that file already exists). 
#-o option takes 1 option, the filename to output the results to. 
#-n option specifies the number of trials to run. default is 5.
# Values for -s option, and what each selects:
# 1) PyCPU
# 2) C++CPU
# 3) PyCUDA
# 4) CUDA

TRIALS=2
SIZE_START=32
SIZE_END=2048
SIZE_STEP=64

SELECTION_REGEX='^[1-4]$';

NUM_TRIALS_REGEX='^[0-9]+$';

FORCE_OVERWRITE=false;
FILE_WAS_SPECIFIED=false

N=0;#Default to 0 to tell later selection code to prompt
    #for user selection.

while getopts "fo:s:n:" opt; do
    case $opt in
	f)#Do not prompt to overwrite file
	    FORCE_OVERWRITE=true;;
	o)#Use this filename:
	    FILENAME=$OPTARG;
	    FILE_WAS_SPECIFIED=true;;
	s)#Use this option for program selection, only if it is 1,2,3, or 4
	    if [[ $OPTARG =~ $SELECTION_REGEX ]]; then
		N=$OPTARG;
	    else
		echo "Error: Option for -s must be an integer between 1 and 4.";
		exit 1;
	    fi
	    ;;
	n)#Used for number of trials.
	    if [[ $OPTARG =~ $NUM_TRIALS_REGEX ]]; then
		TRIALS=$OPTARG;
	    else
		echo "Error: Option for -n must be a non-negative integer.";
		exit 1;
	    fi
	    ;;
	\?)
	    # echo "Invalid option -$OPTARG";
	    exit 1;;
	:)
	    echo "Option -$OPTARG requires an argument";;
    esac
done

if [ $FILE_WAS_SPECIFIED = false ]; then
    echo -n "Enter output filename: ";
    read FILENAME;
fi

#Check if file already exists:
if [ -a $FILENAME ]; then
    #Check if overwrite flag was specified:
    if [ $FORCE_OVERWRITE != true ]; then 
        #Prompt user if overwrite:
	YN="not yet";
	until [[ "${YN,,}" = "y" || "${YN,,}" = "n" ]]; do
	    echo -n "Do you want to overwrite \"$FILENAME\"? [y/n]:";
 	    read YN;
	done
       #Check user prompt back:
	if [[ "${YN,,}" = "n" ]]; then
	    echo "Abort";
	    exit 0;
	fi
    fi
    echo "\"$FILENAME\" will be overwritten...";
fi

if [ $N == 0 ]; then 
    #Prompt for user selection on program type:
    echo "1) PyCPU";
    echo "2) C++CPU";
    echo "3) PyCUDA";
    echo "4) CUDA";
    #Get the user input:
    SELECTION="bad select";
    while ! [[ $SELECTION =~ $SELECTION_REGEX ]]; do
	echo -n "#? ";
	read SELECTION;
    done
else
    #Use command line specified value:
    SELECTION=$N;
fi

#Pre: One argument must be specified to this function 
#     containing the program to run, e.g. "python mand.py".
#Post: Runs the program with the global args specified at the
#      top of this file, and outputs status as well as 
#      relevant benchmark data to $FILENAME
runBenchmark() {
    echo "Running benchmark, storing output in \"$FILENAME\".";
    echo "Trials per format: $TRIALS" >> $FILENAME;
    local CURR_SIZE=$SIZE_START
    while [ $CURR_SIZE -le $SIZE_END ]; do
	echo "${CURR_SIZE} X ${CURR_SIZE}" >> $FILENAME;
	echo "${CURR_SIZE} X ${CURR_SIZE}";
        #Do the individual trials:
	for trial in `seq $TRIALS`; do
	    echo -n "Trial $trial: ";
	    local PARAMS="$CURR_SIZE";
	    #COMMAND="$TIME_COMMAND $PROG $PARAMS";
	    local COMMAND="$1 $PARAMS";
	    echo "Command to run: $COMMAND";
	    local RESULT=$((time $COMMAND) 2>&1);
	    local GREP_FLOATS=`echo $RESULT | grep -o "[0-9]*m[0-9]*\.[0-9]*"`
	    local TIME=`echo $GREP_FLOATS | grep -o "^[0-9]*m[0-9]*\.[0-9]*"`
#	    local TIME=${RESULT:8:5};#Extract the exact time
	    echo -n "$TIME" >> $FILENAME;
	    echo $TIME;
	    if [ $trial = $TRIALS ]; then
		echo "" >> $FILENAME;
	    else
		echo -n " " >> $FILENAME;
	    fi
	done
	let CURR_SIZE+=SIZE_STEP
    done
}

#ASSERT: N = 1,2,3, or 4
#SETUP FILE HEADER:
MESSAGE=" Benchmark for Matrix Multiplication";
if [ $SELECTION = 1 ]; then #PyCPU
    echo "PyCPU$MESSAGE" > $FILENAME;
    runBenchmark "python mult.py"
elif [ $SELECTION = 2 ]; then #C++CPU
    echo "C++CPU$MESSAGE" > $FILENAME;
    runBenchmark "./mult"
elif [ $SELECTION = 3 ]; then #PyCUDA
    echo "PyCUDA$MESSAGE" > $FILENAME;
    runBenchmark "python gpuMult.py"
elif [ $SELECTION = 4 ]; then #CUDA
    echo "CUDA$MESSAGE" > $FILENAME;
    runBenchmark "./gpuMult"
fi
exit 0;
