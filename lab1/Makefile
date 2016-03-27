EXE = lab1
lab1: main.cu counting.o
	nvcc -std=c++11 -arch=sm_30 -O2 main.cu counting.o -o $(EXE)
counting.o: counting.cu counting.h
	nvcc -std=c++11 -arch=sm_30 -O2 -c counting.cu -o counting.o
