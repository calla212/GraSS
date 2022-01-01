#include "nextdoor.cu"

#include <anyoption.h>

template<class SampleType, typename App>
int appMain(int argc, char* argv[]) {
  AnyOption *opt = new AnyOption();
  opt->addUsage("usage: ");
  opt->addUsage("");
  opt->addUsage("-h --help              Prints this help");
  opt->addUsage("-g --graph-file        File containing graph");
  opt->addUsage("-n --nruns             Number of runs");

  opt->setFlag  ("help",           'h');
  opt->setOption("graph-file",     'g');
  opt->setOption("nruns",          'n');

  opt->processCommandArgs(argc, argv);

  if (!opt->hasOptions()) {
    opt->printUsage();
    delete opt;
    return 0;
  }

  char* graph_file = opt->getValue('g');
  char* run_num_str = opt->getValue('n');
  int run_num = 1;

  if (graph_file == nullptr) {
    opt->printUsage();
    delete opt;
    return 0;
  }

  if (run_num_str != nullptr) run_num = atoi(run_num_str);

  nextdoor<SampleType, App>(opt->getValue('g'), run_num);

  return 0;
}