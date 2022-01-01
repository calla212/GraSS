#include <libNextDoor.hpp>
#include <main.cu>
#include "khop.cu"

typedef KHopApp KHopSampling;

//Declare NextDoorData
static NextDoorData<KHopSample, KHopSampling> nextDoorData;

int main(int argc, char* argv[]) {
  //Call appMain to run as a standalone executable
  return appMain<KHopSample, KHopSampling>(argc, argv);
}
