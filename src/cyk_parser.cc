#include "cyk_parser.h"

CykParser::CykParser(const unsigned nt_count, const vector<float>& probs) : vocab_size(probs.size() / nt_count - nt_count * nt_count), nt_count(nt_count), probs(probs) {
  assert (probs.size() % nt_count == 0);
  assert (probs.size() / nt_count > nt_count * nt_count);
}

SyntaxTree CykParser::BackTrace(const vector<vector<vector<BackPointer>>>& table, unsigned i, unsigned j, unsigned n) {
  const unsigned rules_per_nt = nt_count * nt_count + vocab_size;

  BackPointer bp = table[i][j][n];
  assert (bp.rule_index / rules_per_nt == n);

  SyntaxTree r;
  r.label_ = n;
  if (bp.rule_index % rules_per_nt < nt_count * nt_count) {
    assert (bp.split_point > i);
    assert ( bp.split_point < j);
    unsigned rhs1 = (bp.rule_index % rules_per_nt) / nt_count;
    unsigned rhs2 = (bp.rule_index % rules_per_nt) % nt_count;
    SyntaxTree r1 = BackTrace(table, i, bp.split_point, rhs1);
    SyntaxTree r2 = BackTrace(table, bp.split_point, j, rhs2);
    r.children.push_back(r1);
    r.children.push_back(r2);
    return r;
  }
  else {
    unsigned rhs = (bp.rule_index % rules_per_nt) - nt_count * nt_count;
    SyntaxTree r1;
    r1.label_ = rhs;
    r.children.push_back(r1);
    return r;
  }
}

SyntaxTree CykParser::Parse(const Sentence& sentence) {
  const unsigned L = sentence.size();
  const unsigned rules_per_nt = nt_count * nt_count + vocab_size;
  vector<vector<vector<BackPointer>>> table(L, vector<vector<BackPointer>>(L + 1, vector<BackPointer>(nt_count)));
  for (unsigned i = 0; i < L; ++i) {
    for (unsigned n = 0; n < nt_count; ++n) {
      assert ((unsigned)sentence[i] < vocab_size);
      unsigned rule_index = n * rules_per_nt + nt_count * nt_count + sentence[i];
      float score = probs[rule_index];
      assert (i < table.size());
      assert (i + 1 < table[i].size());
      assert (n < table[i][i + 1].size());
      table[i][i + 1][n] = {score, ~0U, rule_index};
    }
  }

  for (unsigned len = 2; len <= L; ++len) {
    for (unsigned i = 0; i < L - len + 1; ++i) {
      for (unsigned j = i + 2; j <= L; ++j) {
        for (unsigned k = i + 1; k < j; ++k) {
          for (unsigned A = 0; A < nt_count; ++A) {
            for (unsigned B = 0; B < nt_count; ++B) {
              for (unsigned C = 0; C < nt_count; ++C) {
                unsigned rule_index = A * rules_per_nt + B * nt_count + C;
                float new_score = table[i][k][B].score + table[k][j][C].score + probs[rule_index];
                if (table[i][j][A].split_point == 0 || new_score > table[i][j][A].score) {
                  table[i][j][A] = {new_score, k, rule_index};
                }
              }
            }
          }
        }
      }
    }
  }

  return BackTrace(table, 0, L, 0);
}

/*KBestList<shared_ptr<SyntaxTree>> CykParser::Parse(const Sentence& sentence, unsigned K) {
  const unsigned L = sentence.size();
  cerr << "Parsing a sentence of length " << L << endl;
  const unsigned vocab_size = probs.size() / nt_count - nt_count * nt_count;
  cerr << "Computed vocab size: " << vocab_size << endl;
  const unsigned rules_per_nt = nt_count * nt_count + vocab_size;

  vector<vector<vector<KBestList<BackPointer>>>> table(L);
  for (unsigned i = 0; i < L; ++i) {
    table[i] = vector<vector<KBestList<BackPointer>>>(L + 1);
    for (unsigned j = 0; j < L + 1; j++) {
      table[i][j] = vector<KBestList<BackPointer>>(nt_count);
      for (unsigned k = 0; k < nt_count; ++k) {
        table[i][j][k] = KBestList<BackPointer>(K);
      }
    }
  }

  for (unsigned i = 0; i < L; ++i) {
    for (unsigned n = 0; n < nt_count; ++n) {
      assert ((unsigned)sentence[i] < vocab_size);
      unsigned rule_index = n * rules_per_nt + nt_count * nt_count + sentence[i];
      cerr << i << ": " << n << " --> " << sentence[i] << " ==> " << rule_index << " (of " << probs.size() << ")" << endl;
      float score = probs[rule_index];
      assert (i < table.size());
      assert (i + 1 < table[i].size());
      assert (n < table[i][i + 1].size());
      table[i][i + 1][n].add(score, {~0U, rule_index});
    }
  }

  for (unsigned len = 2; len <= L; ++len) {
    for (unsigned i = 0; i < L - len + 1; ++i) {
      for (unsigned j = i + 2; j <= L; ++j) {
        for (unsigned k = i + 1; k < j; ++k) {
          for (unsigned A = 0; A < nt_count; ++A) {
            for (unsigned B = 0; B < nt_count; ++B) {
              for (unsigned C = 0; C < nt_count; ++C) {
                unsigned rule_index = A * rules_per_nt + B * nt_count + C;
                for (auto& left : table[i][k][B].hypothesis_list()) {
                  float left_score = left.first;
                  for (auto& right : table[k][j][C].hypothesis_list()) {
                    float right_score = right.first;
                    float score = probs[rule_index] + left_score + right_score;
                    table[i][j][A].add(score, {k, rule_index});
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  KBestList<shared_ptr<SyntaxTree>> r(K);
  return r;
}*/
