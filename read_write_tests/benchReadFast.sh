#!/bin/bash

OUTPUT="benchReadFastTime.txt"
MIN=0
MAX=33553920
STEP=262144
TRIALS=2

echo "read fast to global benchmark" > $OUTPUT

for i in `seq $MIN $STEP $MAX`; do
    echo $i
    echo $i >> $OUTPUT
    for t in `seq $TRIALS`; do
	COMMAND="./readGlobal $i"
	echo "Command: $COMMAND"
	RESULT=`$COMMAND`
	echo -n $RESULT
	echo -n $RESULT >> $OUTPUT
	if [ $t = $TRIALS ]; then
	    echo ''
	    echo '' >> $OUTPUT
	else
	    echo -n ' '
	    echo -n ' ' >> $OUTPUT
	fi
    done
done

exit 0;
