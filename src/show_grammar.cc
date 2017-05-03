#include <iostream>
#include <csignal>
#include "train.h"
#include "pcfglm.h"
#include "utils.h"
#include "io.h"

using namespace dynet;
using namespace dynet::expr;
using namespace dynet::mp;
using namespace std;
namespace po = boost::program_options;

int main(int argc, char** argv) {
  dynet::initialize(argc, argv, true);

  po::options_description desc("description");
  desc.add_options()
  ("help", "Display this help message")
  ("model", po::value<string>(), "Trained model whose grammar will be dumped");

  AddTrainerOptions(desc);

  po::positional_options_description positional_options;
  positional_options.add("model", 1);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).positional(positional_options).run(), vm);

  if (vm.count("help")) {
    cerr << desc;
    return 1;
  }

  po::notify(vm);

  Dict vocab;
  Model dynet_model;
  PcfgLm* model = new PcfgLm();
  Trainer* trainer = nullptr;

  const string model_filename = vm["model"].as<string>();
  Deserialize(model_filename, vocab, *model, dynet_model, trainer);

  ComputationGraph cg;
  model->NewGraph(cg);

  vector<float> probs = as_vector(model->GetRuleProbs().value());
  assert (probs.size() == model->num_rules());
  for (unsigned i = 0; i < model->num_rules(); ++i) {
    cout << model->rule_string(i, vocab) << " ||| " << probs[i] << endl;
  }

  return 0;
}
