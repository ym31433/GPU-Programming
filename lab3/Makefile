EXE = lab3
all:$(EXE)
	rm *.o
	./$(EXE) img_background.ppm img_target.ppm img_mask.pgm 130 600 output.ppm
$(EXE): main.cu $(EXE).o pgm.o
	nvcc -std=c++11 -arch=sm_30 -O2 main.cu $(EXE).o pgm.o -o $(EXE)
$(EXE).o: $(EXE).cu $(EXE).h
	nvcc -std=c++11 -arch=sm_30 -O2 -c $(EXE).cu -o $(EXE).o
pgm.o: pgm.cpp pgm.h
	nvcc -std=c++11 -arch=sm_30 -O2 -c pgm.cpp -o pgm.o
clean:
	rm *.o
