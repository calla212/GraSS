include ../../../common.mk

all: bin

bin: main.cu
	nvcc main.cu -I../../../AnyOption/ ../../../AnyOption/anyoption.cpp -I../../  -lcurand -lpthread  \
	$(ARCH_CODE_FLAGS) -Xcompiler -fopenmp -O3 -o MultiRWSampling -lcurand -Xptxas -O3

clean:
	rm MultiRWSampling