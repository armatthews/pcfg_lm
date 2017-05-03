#pragma once
#include <boost/serialization/access.hpp>
#include "dynet/dynet.h"
#include "utils.h"
#include "mlp.h"

class PcfgLm {
public:
  PcfgLm();
  PcfgLm(unsigned nt_count, unsigned vocab_size, unsigned rule_emb_dim, unsigned hidden_size, Model& dynet_model);
  void SetDropout(float rate);
  void NewGraph(ComputationGraph& cg);

  Expression BuildGraph(const Sentence& sentence);
  Expression GetRuleProbs();
  Expression GetSentenceProb(const Sentence& sentence);

  string nt_string(unsigned i) const;
  string rule_string(unsigned i, Dict& vocab) const;
  unsigned num_rules() const;

private:
  unsigned nt_count;
  unsigned vocab_size;
  LookupParameter rules;
  MLP score_mlp;

  Expression rule_embs;
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
