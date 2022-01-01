include ../../../common.mk

all: bin

bin: DeepWalkBin

DeepWalkBin: main.cu
	nvcc deepwalk.cu -I../../../AnyOption/ ../../../AnyOption/anyoption.cpp -I../../  -lcurand -lpthread  \
	$(ARCH_CODE_FLAGS) -O3 -Xcompiler -fopenmp -o DeepWalkSampling -lcurand -Xptxas -O3

clean:
	rm DeepWalkSampling
