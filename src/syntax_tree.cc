#include <cassert>
#include <iostream>
#include <sstream>
#include "syntax_tree.h"

SyntaxTree::SyntaxTree() : label_(-1), id_(-1) {}

bool SyntaxTree::IsTerminal() const {
  return children.size() == 0;
}

unsigned SyntaxTree::NumChildren() const {
  return children.size();
}

unsigned SyntaxTree::NumNodes() const {
  unsigned node_count = 1;
  for (const SyntaxTree& child : children) {
    node_count += child.NumNodes();
  }
  return node_count;
}

unsigned SyntaxTree::MaxBranchCount() const {
  unsigned max_branch_count = children.size();
  for (const SyntaxTree& child : children) {
    unsigned n = child.MaxBranchCount();
    if (n > max_branch_count) {
      max_branch_count = n;
    }
  }
  return max_branch_count;
}

unsigned SyntaxTree::MinDepth() const {
  if (IsTerminal()) {
    return 0;
  }

  unsigned min_depth = children[0].MinDepth();
  for (unsigned i = 1; i < children.size(); ++i) {
    unsigned d = children[i].MinDepth();
    if (d < min_depth) {
      min_depth = d;
    }
  }

  return min_depth + 1;
}

unsigned SyntaxTree::MaxDepth() const {
  if (IsTerminal()) {
    return 0;
  }

  unsigned max_depth = children[0].MaxDepth();
  for (unsigned i = 1; i < children.size(); ++i) {
    unsigned d = children[i].MaxDepth();
    if (d > max_depth) {
      max_depth = d;
    }
  }

  return max_depth + 1;
}

SyntaxTree& SyntaxTree::GetChild(unsigned i) {
  assert (i < children.size());
  return children[i];
}

const SyntaxTree& SyntaxTree::GetChild(unsigned i) const {
  assert (i < children.size());
  return children[i];
}

WordId SyntaxTree::label() const {
  return label_;
}

unsigned SyntaxTree::id() const {
  return id_;
}

Sentence SyntaxTree::GetTerminals() const {
  Sentence terminals;
  if (IsTerminal()) {
    terminals.push_back(label_);
    return terminals;
  }
  else {
    for (const SyntaxTree& child : children) {
      Sentence child_terminals = child.GetTerminals();
      terminals.insert(terminals.end(), child_terminals.begin(), child_terminals.end());
    }
    return terminals;
  }
}

string SyntaxTree::ToString(Dict& word_dict, Dict& label_dict) const {
  if (IsTerminal()) {
    return word_dict.convert(label_);
  }

  stringstream ss;
  ss << "(" << label_dict.convert(label_);
  for (const SyntaxTree& child : children) {
    ss << " " << child.ToString(word_dict, label_dict);
  }
  ss << ")";
  return ss.str();
}

unsigned SyntaxTree::AssignNodeIds(unsigned start) {
  for (SyntaxTree& child : children) {
    start = child.AssignNodeIds(start);
  }
  id_ = start;
  return start + 1;
}

SyntaxTreeIterator SyntaxTree::begin(TreeIterationOrder order) const {
  SyntaxTree* nonconst_this = const_cast<SyntaxTree*>(this);
  assert (nonconst_this != nullptr);
  return SyntaxTreeIterator(nonconst_this, order);
}

SyntaxTreeIterator SyntaxTree::end() const {
  return SyntaxTreeIterator(nullptr, PreOrder); // order doesn't matter
}

/*ostream& operator<< (ostream& stream, const SyntaxTree& tree) {
  return stream << tree.ToString();
}*/

SyntaxTreeIterator::SyntaxTreeIterator(SyntaxTree* root, TreeIterationOrder order) {
  this->node = root;
  this->order = order;
  if (root == nullptr) {
    return;
  }

  if (order == PreOrder) {
    node_stack.push(root);
    index_stack.push(0);
  }
  else if (order == PostOrder) {
    while (node->NumChildren() > 0) {
      node_stack.push(node);
      index_stack.push(0);
      node = &node->GetChild(0);
    }
    node_stack.push(node);
    index_stack.push(0);
  }
  else {
    assert (false && "Invalid tree iteration order!");
  }
}

SyntaxTree& SyntaxTreeIterator::operator*() {
  return *node;
}

bool SyntaxTreeIterator::operator==(const SyntaxTreeIterator& other) {
  return node == other.node;
}

bool SyntaxTreeIterator::operator!=(const SyntaxTreeIterator& other) {
  return !(*this == other);
}

SyntaxTreeIterator& SyntaxTreeIterator::operator++() {
  assert (node_stack.size() > 0);
  assert (node_stack.size() == index_stack.size());

  SyntaxTree* node = node_stack.top();
  unsigned i = index_stack.top();
  index_stack.pop();

  if (order == PreOrder) {
    while (node_stack.size() > 0 && i >= node->NumChildren()) {
      node_stack.pop();
      if (node_stack.size() > 0) {
        node = node_stack.top();
        i = index_stack.top();
        index_stack.pop();
      }
    }

    if (node_stack.size() > 0) {
      node_stack.push(&node->GetChild(i));
      index_stack.push(i + 1);
      index_stack.push(0);
      this->node = node_stack.top();
    }
    else {
      this->node = nullptr;
    }
  }
  else if (order == PostOrder) {
    assert (node == this->node);
    assert (i >= node->NumChildren());

    node_stack.pop();
    if (node_stack.size() > 0) {
      node = node_stack.top();
      i = index_stack.top();
      index_stack.pop();

      i += 1;
      index_stack.push(i);

      while (i < node->NumChildren()) {
        node = &node->GetChild(i);
        node_stack.push(node);
        index_stack.push(0);
        i = 0;
      }
      this->node = node;
    }
    else {
      this->node = nullptr;
    }
  }
  else {
    assert (false && "Invalid tree iteration order!");
  }

  assert (node_stack.size() == index_stack.size());
  return *this;
}

unsigned SyntaxTree::size() const {
  return NumNodes();
}
