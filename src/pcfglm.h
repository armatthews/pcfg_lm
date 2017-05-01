#pragma once
#include <boost/serialization/access.hpp>
#include "dynet/dynet.h"
#include "utils.h"
#include "mlp.h"

class PcfgLm {
public:
  PcfgLm();
  PcfgLm(unsigned nt_count, unsigned vocab_size, unsigned hidden_size, Model& dynet_model);
  void SetDropout(float rate) {}
  void NewGraph(ComputationGraph& cg);

  Expression BuildGraph(const Sentence& sentence);
  Expression GetRuleProbs();
  Expression GetSentenceProb(const Sentence& sentence);

private:
  unsigned nt_count;
  unsigned vocab_size;
  LookupParameter rules;
  MLP score_mlp;

  ComputationGraph* pcg;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar & nt_count;
    ar & vocab_size;
    ar & rules;
    ar & score_mlp;
  }
};
