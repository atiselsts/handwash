#!/bin/bash

date

for i in 1 2 3
do
#	echo running 224x224 experiment $i
#	./input-shape-is-washing-classifier.py 224 224
        date
done


for i in 1 2 3
do
#	echo running 240x320 experiment $i
#	./input-shape-is-washing-classifier.py 240 320
	date
done

#sleep 7200

date
for i in 1 2 3
do
	echo running 320x240 experiment $i
	./input-shape-is-washing-classifier.py 320 240
	date
done


for i in 1 2 3
do
	echo running 80x60 experiment $i
	./input-shape-is-washing-classifier.py 80 60
	date
done


for i in 1 2 3
do
	echo running 40x30 experiment $i
	./input-shape-is-washing-classifier.py 40 30
	date
done
