#include <iostream>
#include <csignal>
#include <boost/program_options.hpp>
#include "cyk_parser.h"
#include "pcfglm.h"
#include "utils.h"
#include "io.h"

using namespace dynet;
using namespace dynet::expr;
using namespace std;
namespace po = boost::program_options;

void OutputKBestList(unsigned sentence_number, KBestList<shared_ptr<SyntaxTree>> kbest, Dict& vocab) {
}

int main(int argc, char** argv) {
  dynet::initialize(argc, argv, true);

  po::options_description desc("description");
  desc.add_options()
  ("help", "Display this help message")
  ("model", po::value<string>(), "Trained model")
  ("text", po::value<string>(), "Input text file");

  po::positional_options_description positional_options;
  positional_options.add("model", 1);
  positional_options.add("text", 1);

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
  const string text_filename = vm["text"].as<string>();
  Deserialize(model_filename, vocab, *model, dynet_model, trainer);

  vector<Sentence> input_text = ReadText(text_filename, vocab);

  ComputationGraph cg;
  model->NewGraph(cg);

  vector<float> probs = as_vector(model->GetRuleProbs().value());
  CykParser parser(model->nt_count, probs);

  Dict label_vocab;
  for (unsigned i = 0; i < model->nt_count; ++i) {
    label_vocab.convert(((string)"") + (char)('A' + i));
  }

  for (unsigned i = 0; i < input_text.size(); ++i) {
    Sentence& sentence = input_text[i];
    //KBestList<shared_ptr<SyntaxTree>> parse_trees = parser.Parse(sentence);
    //OutputKBestList(i, parse_trees, vocab);
    SyntaxTree tree = parser.Parse(sentence);
    cout << tree.ToString(vocab, label_vocab) << endl;
  }

  return 0;
}
