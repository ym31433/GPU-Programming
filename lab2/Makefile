EXE = lab2
all:$(EXE)
	rm result.mkv
	./lab2
	rm *.o
	avconv -i result.y4m result.mkv
$(EXE): main.cu lab2.o
	nvcc -std=c++11 -arch=sm_30 -O2 --use_fast_math main.cu lab2.o -o $(EXE)
lab2.o: lab2.cu lab2.h
	nvcc -std=c++11 -arch=sm_30 -O2 -c lab2.cu -o lab2.o
clean:
	rm result.mkv
	rm *.o
