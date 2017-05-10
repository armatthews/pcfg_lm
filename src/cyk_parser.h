#pragma once
#include <vector>
#include <memory>
#include "kbestlist.h"
#include "syntax_tree.h"
#include "utils.h"

struct BackPointer {
  float score;
  unsigned split_point;
  unsigned rule_index;
};

class CykParser {
public:
  CykParser(const unsigned nt_count, const vector<float>& probs);
  SyntaxTree BackTrace(const vector<vector<vector<BackPointer>>>& table, unsigned i, unsigned j, unsigned n);
  SyntaxTree Parse(const Sentence& sentence);
  KBestList<shared_ptr<SyntaxTree>> Parse(const Sentence& sentence, unsigned K);
private:
  const unsigned vocab_size;
  const unsigned nt_count;
  const vector<float>& probs;
};

