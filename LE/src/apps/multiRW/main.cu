#include <lib.hpp>
#include <main.cu>
#include "multiRW.cu"

typedef MultiRWApp MultiRWSampling;

//Declare NextDoorData
static NextDoorData<MultiRWSample, MultiRWSampling> nextDoorData;

int main(int argc, char* argv[]) {
  //Call appMain to run as a standalone executable
  return appMain<MultiRWSample, MultiRWSampling>(argc, argv);
}
