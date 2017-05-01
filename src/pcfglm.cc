#include "pcfglm.h"

PcfgLm::PcfgLM(unsigned nt_count, unsigned vocab_size, unsigned hidden_size, Model& model) :
    nt_count(nt_count), vocab_size(vocab_size) {
  unsigned rule_count = nt_count * (nt_count * nt_count + vocab_size);
  rules = model.add_lookup_parameters(rule_count, rule_emb_dim);
  score_mlp = MLP(rule_emb_dim, hidden_size, 1);
}
