#!/usr/bin/env sh

# Martin Kersner, m.kersner@gmail.com
# 2016/04/12

COLS=`tput cols`
ROWS=`tput lines`
CMD="feedgnuplot --lines --points --terminal 'dumb ${COLS},${ROWS}' --exit"
eval $CMD
