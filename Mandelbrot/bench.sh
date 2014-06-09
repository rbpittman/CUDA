#!/bin/bash
#-f option forces overwrite of whichever filename is 
#    specified (if that file already exists). 
#-o option takes 1 option, the filename to output the results to. 
# Values for -n option, and what each selects:
# 1) PyCPU
# 2) C++CPU
# 3) PyCUDA
# 4) CUDA

CPU_TRIALS=5
CPU_IMAGE_START=100
CPU_IMAGE_END=2000
CPU_IMAGE_STEP=100
CPU_IMAGE_DEPTH=500

SELECTION_REGEX='^[1-4]$';

FORCE_OVERWRITE=false;
FILE_WAS_SPECIFIED=false

N=0;#Default to 0 to tell later selection code to prompt
    #for user selection.

while getopts "fo:n:" opt; do
    case $opt in
	f)#Do not prompt to overwrite file
	    FORCE_OVERWRITE=true;;
	o)#Use this filename:
	    FILENAME=$OPTARG;
	    FILE_WAS_SPECIFIED=true;;
	n)#Use this option for program selection, only if it is 1,2,3, or 4
	    if [[ $OPTARG =~ $SELECTION_REGEX ]]; then
		N=$OPTARG;
	    else
		echo "Error: Option for -n must be an integer between 1 and 4.";
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
    echo "Trials per format: $CPU_TRIALS" >> $FILENAME;
    local CURR_IMAGE=$CPU_IMAGE_START
    while [ $CURR_IMAGE -le $CPU_IMAGE_END ]; do
	echo "${CURR_IMAGE} X ${CURR_IMAGE}, depth=$CPU_IMAGE_DEPTH" >> $FILENAME;
	echo "${CURR_IMAGE} X ${CURR_IMAGE}, depth=$CPU_IMAGE_DEPTH";
        #Do the individual trials:
	for trial in `seq $CPU_TRIALS`; do
	    echo -n "Trial $trial: ";
	    local PARAMS="$CURR_IMAGE $CURR_IMAGE $CPU_IMAGE_DEPTH";
	    #COMMAND="$TIME_COMMAND $PROG $PARAMS";
	    local COMMAND="$1 $PARAMS";
	    echo "Command to run: $COMMAND";
	    local RESULT=$((time $COMMAND) 2>&1);
	    local GREP_FLOATS=`echo $RESULT | grep -o "[0-9]*m[0-9]*\.[0-9]*"`
	    local TIME=`echo $GREP_FLOATS | grep -o "^[0-9]*m[0-9]*\.[0-9]*"`
#	    local TIME=${RESULT:8:5};#Extract the exact time
	    echo -n "$TIME" >> $FILENAME;
	    echo $TIME;
	    if [ $trial = $CPU_TRIALS ]; then
		echo "" >> $FILENAME;
	    else
		echo -n " " >> $FILENAME;
	    fi
	done
	let CURR_IMAGE+=CPU_IMAGE_STEP
    done
}

#ASSERT: N = 1,2,3, or 4
#SETUP FILE HEADER:
if [ $SELECTION = 1 ]; then #PyCPU
    echo "PyCPU Benchmark" > $FILENAME;
    runBenchmark "python mand.py"
elif [ $SELECTION = 2 ]; then #C++CPU
    echo "C++CPU Benchmark" > $FILENAME;
    runBenchmark "./mand"
elif [ $SELECTION = 3 ]; then #PyCUDA
    echo "PyCUDA Benchmark" > $FILENAME;
    runBenchmark "./python gpuMand.py"
elif [ $SELECTION = 4 ]; then #CUDA
    echo "CUDA Benchmark" > $FILENAME;
    runBenchmark "./gpuMand"
fi

exit 0;
