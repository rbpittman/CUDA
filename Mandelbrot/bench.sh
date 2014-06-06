#!/bin/bash
CPU_TRIALS=5
CPU_IMAGE_START=500
CPU_IMAGE_END=1000
CPU_IMAGE_STEP=100
CPU_IMAGE_DEPTH=500

TIME_COMMAND="/usr/bin/time -f %e";

if [ -z $1 ]; then
    echo -n "Enter output filename: ";
    read FILENAME;
else
    FILENAME=$1
fi

#Check if file already exists:
if [ -a $FILENAME ]; then
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
    echo "\"$FILENAME\" will be overwritten...";
fi

OPTS="PyCPU C++CPU PyCUDA CUDA";
select opt in $OPTS; do
    echo "Running benchmark, storing output in \"$FILENAME\".";
    
    echo "$opt:" > $FILENAME

    CURR_IMAGE=$CPU_IMAGE_START

    while [ $CURR_IMAGE -le $CPU_IMAGE_END ]; do
	echo "${CURR_IMAGE} X ${CURR_IMAGE}, depth=$CPU_IMAGE_DEPTH" >> $FILENAME;
	echo "${CURR_IMAGE} X ${CURR_IMAGE}, depth=$CPU_IMAGE_DEPTH";
        #Do the individual trials:
	for trial in `seq $CPU_TRIALS`; do
	    echo "Trial $trial" >> $FILENAME;
	    echo -n "Trial $trial: ";
	    PARAMS="$CURR_IMAGE $CURR_IMAGE $CPU_IMAGE_DEPTH";
	    if [ "$opt" = "PyCPU" ]; then
		PROG="python mand.py";
	    elif [ "$opt" = "PyCUDA" ]; then
		PROG="python gpuMand.py";
	    elif [ $opt="C++CPU" ]; then
		PROG="./mand";
	    elif [ $opt="CUDA" ]; then
		PROG="./gpuMand";
	    fi
	    COMMAND="$TIME_COMMAND $PROG $PARAMS";
	    echo $COMMAND;
	    RESULT=$(($COMMAND) 2>&1);
	    echo $RESULT >> $FILENAME;
	    echo $RESULT;
	done
	let CURR_IMAGE+=CPU_IMAGE_STEP
    done
    exit 0;
done #For select
