# GraSS

GraSorw is a disk-based graph processing system designed for scalable second-order random walk tasks. 


## Environment and Dependency
This code is developed and tested on:
* Ubuntu 20.04
* RTX6000 with Nvidia Drive 465.31
* CUDA 11.3
* [CUB](https://nvlabs.github.io/cub/)

## Quick Start

### Dataset preparation

To download the test dataset, in the project root folder, do:

```
mkdir datasets
cd datasets
wget https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz
gzip -d soc-LiveJournal1.txt.gz
cd ..
```

Use the following command to generate an undirected version of this graph:

```
build/Convert2Undir datasets/soc-LiveJournal1.txt
```


### Build
To build this, use ```make``` in folder `LE` or `GT`.

The executable file should be generated in folder `bin` 

### Compress
Enter the folder `chunk_compress` of `LE` or `GT`. Then run ```make``` to build the compressor.

Two executable files should be generated in this folder.

```graph_sort``` is used to generate an undirected version of the graph.

```./graph_sort <input_path> <output_path> [<skip_lines>]```

```compressor``` is used to compress the graph.

```./compressor <input_path> <output_path>```

Two files `*.offset` `*.graph` are generated.

### Execution
To run the sampling, enter the `bin` folder.

```./app -g <cgc_path> [-n <run_times>]```

For example, if the **sample.cgr.graph** and **sample.cgr.offset** are located in DIR_PATH, then execute for random walk

```./rw DIR_PATH/sample.cgr```.

The parameter <run_time> is default as 1.

The other parameters about the graph sampling can be modified in the source code in `./src/app/*`.
