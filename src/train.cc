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

class Learner : public ILearner<Sentence, SufficientStats> {
public:
  Learner(Dict& vocab, PcfgLm& model, Model& dynet_model, Trainer* trainer) : vocab(vocab), model(model), dynet_model(dynet_model), trainer(trainer) {}
  ~Learner() {}
  SufficientStats LearnFromDatum(const Sentence& datum, bool learn) {
    ComputationGraph cg;
    model.NewGraph(cg);

    if (learn) {
      model.SetDropout(dropout_rate);
    }
    else {
      model.SetDropout(0.0f);
    }

    Expression loss_expr = model.BuildGraph(datum);
    dynet::real loss = as_scalar(cg.forward(loss_expr));
    if (learn) {
      cg.backward(loss_expr);
    }
    return SufficientStats(loss, datum.size(), 1);
  }

  void SaveModel() {
    if (!quiet) {
      Serialize(vocab, model, dynet_model, trainer);
    }
  }

  bool quiet;
  float dropout_rate;
private:
  Dict& vocab;
  PcfgLm& model;
  Model& dynet_model;
  Trainer* trainer;
};

// This function lets us elegantly handle the user pressing ctrl-c.
// We set a global flag, which causes the training loops to clean up
// and break. In particular, this allows models to be saved to disk
// before actually exiting the program.
bool ctrlc_pressed = false;
void ctrlc_handler(int signal) {
  if (ctrlc_pressed) {
    cerr << "Exiting..." << endl;
    exit(1);
  }
  else {
    cerr << "Ctrl-c pressed!" << endl;
    ctrlc_pressed = true;
    dynet::mp::stop_requested = true;
  }
}

vector<Sentence> ReadText(const string& filename, Dict& vocab) {
  ifstream f(filename);
  if (!f.is_open()) {
    cerr << "Unable to open " << filename << " for reading." << endl;
    exit(1);
  }

  vector<Sentence> r;
  for (string line; getline(f, line);) {
    r.push_back(read_sentence(line, vocab));
  }

  return r;
}

int main(int argc, char** argv) {
  signal (SIGINT, ctrlc_handler);
  cerr << "Invoked as:";
  for (int i = 0; i < argc; ++i) {
    cerr << " " << argv[i];
  }
  cerr << "\n";

  dynet::initialize(argc, argv, true);

  po::options_description desc("description");
  desc.add_options()
  ("help", "Display this help message")
  ("train_text", po::value<string>()->required(), "Training text")
  ("dev_text", po::value<string>()->required(), "Dev text, used for early stopping")
  ("num_iterations,i", po::value<unsigned>()->default_value(UINT_MAX), "Number of epochs to train for")
  ("cores,j", po::value<unsigned>()->default_value(1), "Number of CPU cores to use for training")
  ("dropout_rate", po::value<float>()->default_value(0.0), "Dropout rate (should be >= 0.0 and < 1)")
  ("report_frequency,r", po::value<unsigned>()->default_value(100), "Show the training loss of every r examples")
  ("dev_frequency,d", po::value<unsigned>()->default_value(10000), "Run the dev set every d examples. Save the model if the score is a new best")
  ("quiet,q", "Do not output model")
  ("model", po::value<string>(), "Reload this model and continue learning");

  AddTrainerOptions(desc);

  po::positional_options_description positional_options;
  positional_options.add("train_text", 1);
  positional_options.add("dev_text", 1);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).positional(positional_options).run(), vm);

  if (vm.count("help")) {
    cerr << desc;
    return 1;
  }

  po::notify(vm);

  const unsigned num_cores = vm["cores"].as<unsigned>();
  const unsigned num_iterations = vm["num_iterations"].as<unsigned>();
  const string train_text_filename = vm["train_text"].as<string>();
  const string dev_text_filename = vm["dev_text"].as<string>();

  Dict vocab;
  Model dynet_model;
  PcfgLm* model = nullptr;
  Trainer* trainer = nullptr;

  if (vm.count("model")) {
    string model_filename = vm["model"].as<string>();
    model = new PcfgLm();
    Deserialize(model_filename, vocab, *model, dynet_model, trainer);
    assert (vocab.is_frozen());

    if (vm.count("sgd") || vm.count("adagrad") || vm.count("adam") || vm.count("rmsprop") || vm.count("momentum")) {
      trainer = CreateTrainer(dynet_model, vm);
    }
  }

  vector<Sentence> train_text = ReadText(train_text_filename, vocab); 

  if (!vm.count("model")) {
    model = new PcfgLm();
    vocab.freeze();
    vocab.set_unk("UNK");
  }

  vector<Sentence> dev_text = ReadText(dev_text_filename, vocab);

  cerr << "Vocabulary size: " << vocab.size() << endl; 
  cerr << "Total parameters: " << dynet_model.parameter_count() << endl;

  trainer = CreateTrainer(dynet_model, vm);
  Learner learner(vocab, *model, dynet_model, trainer);
  learner.quiet = vm.count("quiet") > 0;
  learner.dropout_rate = vm["dropout_rate"].as<float>();
  unsigned dev_frequency = vm["dev_frequency"].as<unsigned>();
  unsigned report_frequency = vm["report_frequency"].as<unsigned>();

  /*if (num_cores > 1) {
    run_multi_process<Sentence>(num_cores, &learner, trainer, train_text, dev_text, num_iterations, dev_frequency, report_frequency);
  }
  else {
    run_single_process<Sentence>(&learner, trainer, train_text, dev_text, num_iterations, dev_frequency, report_frequency, 1);
  }*/

  return 0;
}
