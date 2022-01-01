#include <iostream>
#include <omp.h>
#include "compressor.hpp"
#include <string>

int main(int argc,char *argv[]) {

    if (argc != 3 && argc != 4) {
        printf("incorrect arguments.\n");
        printf("<input_path> <output_path> [<encode_type>]\n");
        abort();
    }
    std::string input_path(argv[1]);
    std::string output_path(argv[2]);
    std::string encode_type = "normal";
    if (argc == 4) encode_type = std::string(argv[3]);

    auto compressor = Compressor(3);
    compressor.load_graph(input_path);

    printf("%s graph loaded.\n", input_path.c_str());

    compressor.compress(encode_type);
    
    compressor.write_cgr(output_path);
    printf("CGR generation completed.\n");

    return 0;
}