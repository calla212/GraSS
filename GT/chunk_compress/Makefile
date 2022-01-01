all:
	g++ compress.cpp compressor.hpp -o compressor -pthread -fopenmp
	g++ graph_sort.cpp -o graph_sort -O3 -std=c++17
compress:
	g++ compress.cpp compressor.hpp -o compressor -pthread -fopenmp
graph_sort:
	g++ graph_sort.cpp -o graph_sort -O3 -std=c++17
clean:
	rm compressor graph_sort
