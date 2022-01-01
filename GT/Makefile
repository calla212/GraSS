include common.mk

MKDIR_P = mkdir -p
BUILD_DIR = ./bin
BUILD_FLAGS =  -O3 -Xptxas -O3 -Xcompiler -fopenmp -lcurand -lpthread
all: tests
directories: 
	${MKDIR_P} $(BUILD_DIR)

tests: directories $(BUILD_DIR)/rw $(BUILD_DIR)/mrw $(BUILD_DIR)/ks

$(BUILD_DIR)/rw: src/apps/randomwalk/main.cu
	nvcc $< $(BUILD_FLAGS) $(ARCH_CODE_FLAGS) -I./AnyOption/ ./AnyOption/anyoption.cpp -I./src -o $@

$(BUILD_DIR)/mrw: src/apps/multiRW/main.cu
	nvcc $< $(BUILD_FLAGS) $(ARCH_CODE_FLAGS) -I./AnyOption/ ./AnyOption/anyoption.cpp -I./src -o $@

$(BUILD_DIR)/ks: src/apps/khop/main.cu
	nvcc $< $(BUILD_FLAGS) $(ARCH_CODE_FLAGS) -I./AnyOption/ ./AnyOption/anyoption.cpp -I./src -o $@

clean:
	rm bin/*
