#include "impl/oracle.h"

#include <cassert>
#include <fstream>
#include <strstream>
#include <algorithm>
#include "dynet/dict.h"
using namespace std;

namespace SRL {


Oracle::~Oracle() {}

inline bool is_ws(char x) { //check whether the character is a space or tab delimiter
  return (x == ' ' || x == '\t');
}

inline bool is_not_ws(char x) {
  return (x != ' ' && x != '\t');
}

inline bool islower(char c){if(c >= 'a' && c <= 'z') return true; return false;}
inline bool isupper(char c){if(c >= 'A' && c <= 'Z') return true; return false;}
inline bool isdigital(char c){if(c >= '0' && c <= '9') return true; return false;}
inline bool isalpha(char c){if(islower(c) || isupper(c)) return true; return false;}

inline std::string unkized(const std::string word, const std::map<string,int>& train_dict){
    int numCaps = 0;
    bool hasDigit = false;
    bool hasDash = false;
    bool hasLower = false;
    
    std::string result = "UNK";
    for(unsigned i = 0; i < word.size(); i ++){
        if(isdigital(word[i])) hasDigit = true;
	else if(word[i] == '-') hasDash = true;
        else if(islower(word[i])) hasLower = true;
	else if(isupper(word[i])) numCaps += 1;
    }
    string lower = word;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower); 
    if(isupper(word[0])){
        if(numCaps == 1) {
	    result = result + "-INITC";
	    if(train_dict.find(lower) != train_dict.end()) result = result + "-KNOWNLC";
	}
	else{
	    result = result + "-CAPS";
	}
    }
    else if(isalpha(word[0]) == false && numCaps > 0)	result = result + "-CAPS";
    else if(hasLower) result = result + "-LC";
    
    if(hasDigit) result = result + "-NUM";
    if(hasDash) result = result + "-DASH";

    
    if(lower.back() == 's' && lower.size() >= 3){
    	char ch2 = lower.size() >= 2 ? lower[lower.size()-2] : '0';
        if(ch2 != 's' && ch2 != 'i' && ch2 != 'u') result = result + "-s";
    }
    else if(lower.size() >= 5 && hasDash == false && (hasDigit == false || numCaps == 0)){
      	char ch1 = lower.size() >= 1 ? lower[lower.size()-1] : '0';
	char ch2 = lower.size() >= 2 ? lower[lower.size()-2] : '0';
        char ch3 = lower.size() >= 3 ? lower[lower.size()-3] : '0';

	if(ch2 == 'e' && ch1 == 'd')
       	    result = result + "-ed";
    	else if(ch3 == 'i' && ch2 == 'n' && ch1 == 'g')
       	    result = result + "-ing";
	else if(ch3 == 'i' && ch2 == 'o' && ch1 == 'n')
            result = result + "-ion";
	else if(ch2 == 'e' && ch1 == 'r')
            result = result + "-er";
	else if(ch3 == 'e' && ch2 == 's' && ch1 == 't')
            result = result + "-est";
	else if(ch2 == 'l' && ch1 == 'y')
            result = result + "-ly";
	else if(ch3 == 'i' && ch2 == 't' && ch1 == 'y')
            result = result + "-ity";
	else if(ch1 == 'y')
            result = result + "-y";
	else if(ch2 == 'a' && ch1 == 'l')
	    result = result + "-al";
    }
    return result;
}

void Oracle::ReadSentenceView(const std::string& line, dynet::Dict* dict, vector<int>* sent) {
  unsigned cur = 0;
  while(cur < line.size()) {
    while(cur < line.size() && is_ws(line[cur])) { ++cur; }
    unsigned start = cur;
    while(cur < line.size() && is_not_ws(line[cur])) { ++cur; }
    unsigned end = cur;
    if (end > start) {
      unsigned x = dict->convert(line.substr(start, end - start));
      sent->push_back(x);
    }
  }
  assert(sent->size() > 0); // empty sentences not allowed
}

void SRLOracle::load_oracle(const string& file, const std::map<string,int>& train_dict) {
  cerr << "Loading top-down oracle from " << file << " ...\n";
  ifstream in(file.c_str());
  assert(in);
  string line;
  vector<int> cur_vs;
  vector<Argument> cur_srs;
  vector<unsigned> index_vs;
  vector<int> cur_ss;
  sents.resize(1);
  while(getline(in, line)) {
    if (line.size() == 0) {
	if (!sents.back().SizesMatch()) {
            cerr << "Mismatched lengths of input strings in oracle before line " << sents.size() << endl;
      	    abort();
        }
        for(unsigned i = 0; i < cur_srs.size(); i ++){
            cur_srs[i].verb = index_vs[cur_srs[i].verb];
        }
	vs.push_back(cur_vs);
        srs.push_back(cur_srs);
	
	/*cerr<<"====================\n";
        for(unsigned i = 0; i < sents.back().raw.size(); ++i){
		cerr<<d->convert(sents.back().raw[i])<<" "<<d->convert(sents.back().lc[i])<<" "<<d->convert(sents.back().unk[i])<<" "<<pd->convert(sents.back().pos[i])<<"\n";
	}
	unsigned j = 0;
	for(unsigned i = 0; i < cur_vs.size(); ++i){
		if(vd->convert(cur_vs[i]) == "NONE") continue;
   		cerr << index_vs[j++]<<":"<<vd->convert(cur_vs[i])<<"\n";
	}
	for(unsigned i = 0; i < cur_srs.size(); ++i){
                cerr << cur_srs[i].argu<<" "<<cur_srs[i].verb<<" "<<ard->convert(cur_srs[i].label)<<"\n";
        }*/
        cur_vs.clear();
        cur_srs.clear();
        index_vs.clear();
    	sents.resize(sents.size() + 1);
	continue;
    }
    auto& cur_sent = sents.back();
    istrstream istr(line.c_str());
    string str;
    unsigned num;
    istr >> num; //19
    istr >> str; //Classics
    cur_sent.raw.push_back(d->convert(str));
    if(train_dict.find(str) == train_dict.end())
    	cur_sent.unk.push_back(d->convert(unkized(str,train_dict)));
    else
	cur_sent.unk.push_back(d->convert(str));
    std::transform(str.begin(), str.end(), str.begin(), ::tolower);
    cur_sent.lc.push_back(d->convert(str));
    istr >> str; //classics
    istr >> str; //classics
    istr >> str; //NNS gold pos
    istr >> str; //NNS pred pos
    cur_sent.pos.push_back(pd->convert(str));
    istr >> str; // _
    istr >> str; // _
    istr >> num; // 20 gold head
    istr >> num; // 20 pred head
    istr >> str; // SBJ gold dependency
    istr >> str; // SBJ pred dependency
   
    istr >> str; // _ or Y verb
    if(str == "Y"){
       	istr >> str; // _ or word.01 kinds
       	cur_vs.push_back(vd->convert(str.substr(str.size()-2, 2)));
	cur_ss.push_back(d->convert(str.substr(0,str.size()-3));
	index_vs.push_back(cur_sent.raw.size()-1);
    }
    else{
       	istr >> str;
        cur_vs.push_back(vd->convert("NONE"));
	cur_ss.push_back(d->convert(""
    }
    unsigned cnt = 0;
    while(istr >> str){
        if(str != "_"){
	    cur_srs.push_back(Argument(cur_sent.raw.size()-1, cnt, ard->convert(str)));
	}
        cnt += 1;
    }
  }
  sents.pop_back(); // delete one more back item, which is defaultly assigned at the end of the reader.
  cerr << "Loaded " << sents.size() << " sentences\n";
  cerr << "    cumulative     verb vocab size: " << vd->size() << endl;
  cerr << "    cumulative terminal vocab size: " << d->size() << endl;
  cerr << "    cumulative argument vocab size: " << ard->size() << endl;
  cerr << "    cumulative      pos vocab size: " << pd->size() << endl;
}
} // namespace srl
