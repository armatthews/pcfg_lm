#include <sstream>
#include "pcfglm.h"
using namespace std;

Dict* g_vocab = nullptr;

PcfgLm::PcfgLm() {}

PcfgLm::PcfgLm(unsigned nt_count, unsigned vocab_size, unsigned rule_emb_dim, unsigned hidden_size, Model& model) :
    nt_count(nt_count), vocab_size(vocab_size), pcg(nullptr) {
  unsigned rule_count = nt_count * (nt_count * nt_count + vocab_size);
  rules = model.add_lookup_parameters(rule_count, {rule_emb_dim});
  score_mlp = MLP(model, rule_emb_dim, hidden_size, 1);
  rule_embs.pg = nullptr;
  cerr << "Created a PCFG-LM with " << rule_count << " rules." << endl;
}

void PcfgLm::SetDropout(float rate) {}

void PcfgLm::NewGraph(ComputationGraph& cg) {
  pcg = &cg;
  rule_embs = parameter(cg, rules);
  score_mlp.NewGraph(cg);
}

Expression PcfgLm::BuildGraph(const Sentence& sentence) {
  return GetSentenceProb(sentence);
}

Expression PcfgLm::GetRuleProbs() {
  const unsigned rules_per_nt = (nt_count * nt_count + vocab_size);

  Expression scores = score_mlp.Feed(rule_embs); // 1xR
  scores = reshape(scores, {nt_count * rules_per_nt}); // R

  vector<Expression> chunks(nt_count);
  for (unsigned i = 0; i < nt_count; ++i) {
    Expression range = pickrange(scores, i * rules_per_nt, (i + 1) * rules_per_nt);
    chunks[i] = softmax(range);
  }

  Expression rule_probs = concatenate(chunks);
  return rule_probs;
}

Expression PcfgLm::GetSentenceProb(const Sentence& sentence) {
  assert (sentence.size() > 0);
  static_assert(std::numeric_limits<float>::is_iec559, "IEEE 754 required");
  const float neg_inf = -std::numeric_limits<float>::infinity();

  // table[i, j, n] gives the probability that the span [i, j) parses as non-terminal N
  // i should be in [0, L], j should be in (i, L], and n should be in [0, nt_count), where L is the length of the sentence
  // Note: half this table (where j <= i) is unused.
  const unsigned L = sentence.size();
  Expression probs = GetRuleProbs();
  vector<vector<vector<Expression>>> table(L, vector<vector<Expression>>(L + 1, vector<Expression>(nt_count, input(*pcg, 0.0f))));

  // Seed the table with pre-terminal probabilities
  const unsigned rules_per_nt = (nt_count * nt_count + vocab_size);
  for (unsigned i = 0; i < L; ++i) {
    for (unsigned n = 0; n < nt_count; ++n) {
      unsigned rule_index = n * rules_per_nt + nt_count * nt_count + sentence[i];
      assert (rule_index < rules_per_nt * nt_count);
      table[i][i + 1][n] = pick(probs, rule_index);
    }
  }

  // Cache all the binary rule probs, since we'll be using them a lot.
  // each rule is of the form A --> B C
  vector<vector<vector<Expression>>> binary_rule_probs(nt_count, vector<vector<Expression>>(nt_count, vector<Expression>(nt_count)));
  for (unsigned A = 0; A < nt_count; ++A) {
    for (unsigned B = 0; B < nt_count; ++B) {
      for (unsigned C = 0; C < nt_count; ++C) {
        unsigned rule_index = A * rules_per_nt + B * nt_count + C;
        assert (rule_index < rules_per_nt * nt_count);
        binary_rule_probs[A][B][C] = pick(probs, rule_index);
      }
    }
  }

  // Now we try to apply all binary rules of the form A --> B C
  // to spans [i, j), with A covering [i, j), B covering [i, k) for
  // some k in (i, j], and C covering [k, j).
  for (unsigned len = 2; len <= L; ++len) {
    for (unsigned i = 0; i <= L - len; ++i) {
      for (unsigned j = i + 1; j <= i + len; ++j) {
        for (unsigned k = i + 1; k < j; ++k) {
          for (unsigned A = 0; A < nt_count; ++A) {
            for (unsigned B = 0; B < nt_count; ++B) {
              for (unsigned C = 0; C < nt_count; ++C) {
                assert (i < k);
                assert (k < j);
                //cerr << i << ", " << k << ", " << j << ", " << A << " --> " << B << " " << C << endl;
                table[i][j][A] = table[i][j][A] + binary_rule_probs[A][B][C] * table[i][k][B] * table[k][j][C];
              }
            }
          }
        }
      }
    }
  }

  return -log(table[0][L][0] + 1e-40f);
}

string PcfgLm::nt_string(unsigned i) const {
  assert (i < nt_count);
  assert (i < 26);
  char r = 'A' + i;
  return string() + r;
}

string PcfgLm::rule_string(unsigned i, Dict& vocab) const {
  assert (i < num_rules());
  const unsigned rules_per_nt = nt_count * nt_count + vocab_size;
  ostringstream oss;
  unsigned lhs = i / rules_per_nt;
  i -= lhs * rules_per_nt;
  oss << nt_string(lhs) << " --> ";

  if (i < nt_count * nt_count) {
    unsigned rhs1 = i / nt_count;
    unsigned rhs2 = i % nt_count;
    oss << nt_string(rhs1) << " " << nt_string(rhs2);
  }
  else {
    i -= nt_count * nt_count;
    oss << vocab.convert(i);
  }
  return oss.str();
}

unsigned PcfgLm::num_rules() const {
  return nt_count * nt_count * nt_count + nt_count * vocab_size;
}
