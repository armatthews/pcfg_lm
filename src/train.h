#pragma once
#include "dynet/dynet.h"
#include "dynet/mp.h"
#include <boost/program_options.hpp>

using namespace dynet;
using namespace dynet::expr;
using namespace dynet::mp;
using namespace std;
namespace po = boost::program_options;

class SufficientStats {
public:
  dynet::real loss;
  unsigned word_count;
  unsigned sentence_count;

  SufficientStats() : loss(), word_count(), sentence_count() {}

  SufficientStats(dynet::real loss, unsigned word_count, unsigned sentence_count) : loss(loss), word_count(word_count), sentence_count(sentence_count) {}

  SufficientStats& operator+=(const SufficientStats& rhs) {
    loss += rhs.loss;
    word_count += rhs.word_count;
    sentence_count += rhs.sentence_count;
    return *this;
  }

  friend SufficientStats operator+(SufficientStats lhs, const SufficientStats& rhs) {
    lhs += rhs;
    return lhs;
  }

  bool operator<(const SufficientStats& rhs) {
    return loss < rhs.loss;
  }

  friend std::ostream& operator<< (std::ostream& stream, const SufficientStats& stats) {
    return stream << exp(stats.loss / stats.word_count) << " (" << stats.loss << " over " << stats.word_count << " words)";
  }
};

void AddTrainerOptions(po::options_description& desc) {
  desc.add_options()
  ("sgd", "Use SGD for optimization")
  ("momentum", po::value<double>(), "Use SGD with this momentum value")
  ("adagrad", "Use Adagrad for optimization")
  ("adadelta", "Use Adadelta for optimization")
  ("rmsprop", "Use RMSProp for optimization")
  ("adam", "Use Adam for optimization")
  ("learning_rate", po::value<double>(), "Learning rate for optimizer (SGD, Adagrad, Adadelta, and RMSProp only)")
  ("alpha", po::value<double>(), "Alpha (Adam only)")
  ("beta1", po::value<double>(), "Beta1 (Adam only)")
  ("beta2", po::value<double>(), "Beta2 (Adam only)")
  ("rho", po::value<double>(), "Moving average decay parameter (RMSProp and Adadelta only)")
  ("epsilon", po::value<double>(), "Epsilon value for optimizer (Adagrad, Adadelta, RMSProp, and Adam only)")
  ("eta_decay", po::value<double>()->default_value(0.05), "Learning rate decay rate (SGD only)")
  ("no_clipping", "Disable clipping of gradients");
}

Trainer* CreateTrainer(Model& dynet_model, const po::variables_map& vm) {
  double eta_decay = vm["eta_decay"].as<double>();
  bool clipping_enabled = (vm.count("no_clipping") == 0);
  unsigned learner_count = vm.count("sgd") + vm.count("momentum") + vm.count("adagrad") + vm.count("adadelta") + vm.count("rmsprop") + vm.count("adam");
  if (learner_count > 1) {
    cerr << "Invalid parameters: Please specify only one learner type.";
    exit(1);
  }

  Trainer* trainer = nullptr;
  if (vm.count("momentum")) {
    double learning_rate = (vm.count("learning_rate")) ? vm["learning_rate"].as<double>() : 0.01;
    double momentum = vm["momentum"].as<double>();
    trainer = new MomentumSGDTrainer(dynet_model, learning_rate, momentum, 0.0);
  }
  else if (vm.count("adagrad")) {
    double learning_rate = (vm.count("learning_rate")) ? vm["learning_rate"].as<double>() : 0.1;
    double eps = (vm.count("epsilon")) ? vm["epsilon"].as<double>() : 1e-20;
    trainer = new AdagradTrainer(dynet_model, learning_rate, eps, 0.0);
  }
  else if (vm.count("adadelta")) {
    double eps = (vm.count("epsilon")) ? vm["epsilon"].as<double>() : 1e-6;
    double rho = (vm.count("rho")) ? vm["rho"].as<double>() : 0.95;
    trainer = new AdadeltaTrainer(dynet_model, eps, rho, 0.0);
  }
  else if (vm.count("rmsprop")) {
    double learning_rate = (vm.count("learning_rate")) ? vm["learning_rate"].as<double>() : 0.1;
    double eps = (vm.count("epsilon")) ? vm["epsilon"].as<double>() : 1e-20;
    double rho = (vm.count("rho")) ? vm["rho"].as<double>() : 0.95;
    trainer = new RMSPropTrainer(dynet_model, learning_rate, eps, rho, 0.0);
  }
  else if (vm.count("adam")) {
    double alpha = (vm.count("alpha")) ? vm["alpha"].as<double>() : 0.001;
    double beta1 = (vm.count("beta1")) ? vm["beta1"].as<double>() : 0.9;
    double beta2 = (vm.count("beta2")) ? vm["beta2"].as<double>() : 0.999;
    double eps = (vm.count("epsilon")) ? vm["epsilon"].as<double>() : 1e-8;
    trainer = new AdamTrainer(dynet_model, alpha, beta1, beta2, eps, 0.0);
  }
  else { /* sgd */
    double learning_rate = (vm.count("learning_rate")) ? vm["learning_rate"].as<double>() : 0.1;
    trainer = new SimpleSGDTrainer(dynet_model, learning_rate, 0.0);
  }
  assert (trainer != nullptr);

  trainer->eta_decay = eta_decay;
  trainer->clipping_enabled = clipping_enabled;
  return trainer;
}
