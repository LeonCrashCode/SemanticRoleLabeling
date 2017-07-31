#ifndef SRL_ORACLE_H_
#define SRL_ORACLE_H_

#include <iostream>
#include <vector>
#include <string>
#include <map>

namespace dynet { class Dict; }

namespace SRL {

// a sentence can be viewed in 4 different ways:
//   raw tokens, UNKed, lowercased, and POS tags
struct Argument {
  unsigned argu;
  unsigned verb;
  int label;
  Argument(){}
  Argument(unsigned a, unsigned v, int l) : argu(a), verb(v), label(l){}
  ~Argument(){}
};

struct Sentence {
  bool SizesMatch() const { return raw.size() == unk.size() && raw.size() == lc.size() && raw.size() == pos.size(); }
  size_t size() const { return raw.size(); }
  std::vector<int> raw, unk, lc, pos;
};

// base class for transition based parse oracles
struct Oracle {
  virtual ~Oracle();
  Oracle(dynet::Dict* dict, dynet::Dict* vdict, dynet::Dict* pdict) : d(dict), vd(vdict), pd(pdict), sents() {}
  unsigned size() const { return sents.size(); }
  dynet::Dict* d;  // dictionary of terminal symbols
  dynet::Dict* vd; // dictionary of action types
  dynet::Dict* pd; // dictionary of POS tags (preterminal symbols)
  std::vector<Sentence> sents;
  std::vector<std::vector<int> > vs;
  std::vector<std::vector<Argument> > srs;
  std::vector<std::vector<int> > ss;
 protected:
  static void ReadSentenceView(const std::string& line, dynet::Dict* dict, std::vector<int>* sent);
};

class SRLOracle : public Oracle {
 public:
  SRLOracle(dynet::Dict* termdict, dynet::Dict* vdict, dynet::Dict* pdict, dynet::Dict* argudict) :
      Oracle(termdict, vdict, pdict), ard(argudict) {}
  // if is_training is true, then both the "raw" tokens and the mapped tokens
  // will be read, and both will be available. if false, then only the mapped
  // tokens will be available
  void load_oracle(const std::string& file, const std::map<std::string, int>& train_dict);
  dynet::Dict* ard; // dictionary of nonterminal types
};

} // namespace parser

#endif
