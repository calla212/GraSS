#include <lib.hpp>
#include <main.cu>
#include "randomWalks.cu"

typedef DeepWalkApp DeepWalkSampling;

static NextDoorData<DummySample, DeepWalkApp> nextDoorData;

int main(int argc, char* argv[]) {
  //Call appMain to run as a standalone executable
  return appMain<DummySample, DeepWalkApp>(argc, argv);
}
