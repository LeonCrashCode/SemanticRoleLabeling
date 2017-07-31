#include <cstdlib>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <vector>
#include <limits>
#include <cmath>
#include <chrono>
#include <ctime>

#include <unordered_map>
#include <unordered_set>

#include <execinfo.h>
#include <unistd.h>
#include <signal.h>

#include "dynet/training.h"
#include "dynet/dynet.h"
#include "dynet/expr.h"
#include "dynet/nodes.h"
#include "dynet/lstm.h"
#include "dynet/rnn.h"
#include "dynet/io.h"
#include "dynet/dict.h"

#include "impl/oracle.h"
#include "impl/cl-args.h"

dynet::Dict termdict, argudict, vdict, posdict;

volatile bool requested_stop = false;
unsigned VERB_SIZE = 0;
unsigned VOCAB_SIZE = 0;
unsigned ARG_SIZE = 0;
unsigned POS_SIZE = 0;

using namespace dynet;
using namespace std;

Params params;
std::map<std::string,int> train_dict;
unordered_map<unsigned, vector<float>> pretrained;
vector<bool> singletons; // used during training
std::map<int, std::vector<unsigned> > possible_senses;
std::vector<unsigned> all_possible_senses;

struct SRLBuilder {

  LSTMBuilder l2rbuilder;
  LSTMBuilder r2lbuilder;
  LookupParameter p_w; // word embeddings
  LookupParameter p_t; // pretrained word embeddings (not updated)
  LookupParameter p_p; // pos tag embeddings
  Parameter p_w2l; // word to LSTM input
  Parameter p_p2l; // POS to LSTM input
  Parameter p_t2l; // pretrained word embeddings to LSTM input
  Parameter p_lb; // LSTM input bias

  Parameter p_sent_start;
  Parameter p_sent_end;

  Parameter p_input2v;
  Parameter p_vbias;
  
  Parameter p_head2argu;
  Parameter p_dep2argu;
  Parameter p_argubias;

  explicit SRLBuilder(Model* model, const unordered_map<unsigned, vector<float>>& pretrained) :
      l2rbuilder(params.layers, params.bilstm_input_dim, params.bilstm_hidden_dim, *model),
      r2lbuilder(params.layers, params.bilstm_input_dim, params.bilstm_hidden_dim, *model),
      p_w(model->add_lookup_parameters(VOCAB_SIZE, {params.input_dim})),
      p_w2l(model->add_parameters({params.bilstm_input_dim, params.input_dim})),
      p_lb(model->add_parameters({params.bilstm_input_dim})),
      p_sent_start(model->add_parameters({params.bilstm_input_dim})),
      p_sent_end(model->add_parameters({params.bilstm_input_dim})),
      p_input2v(model->add_parameters({VERB_SIZE, params.bilstm_hidden_dim*2})),
      p_vbias(model->add_parameters({VERB_SIZE})),
      p_head2argu(model->add_parameters({ARG_SIZE, params.bilstm_hidden_dim*2})),
      p_dep2argu(model->add_parameters({ARG_SIZE, params.bilstm_hidden_dim*2})),
      p_argubias(model->add_parameters({ARG_SIZE})){
    if (params.use_pos) {
      p_p = model->add_lookup_parameters(POS_SIZE, {params.pos_dim});
      p_p2l = model->add_parameters({params.bilstm_input_dim, params.pos_dim});
    }
    if (pretrained.size() > 0) {
      p_t = model->add_lookup_parameters(VOCAB_SIZE, {params.pretrained_dim});
      for (auto it : pretrained)
        p_t.initialize(it.first, it.second);
      p_t2l = model->add_parameters({params.bilstm_input_dim, params.pretrained_dim});
    }
  }

int find(unsigned head, unsigned dep, const vector<SRL::Argument>& sr){
	for(unsigned i = 0; i < sr.size(); i ++){
		if(sr[i].verb == head && sr[i].argu == dep){
			return sr[i].label;
		}
	}
	return argudict.convert("NONE");
}

// *** if correct_actions is empty, this runs greedy decoding ***
// returns parse actions for input sentence (in training just returns the reference)
// OOV handling: raw_sent will have the actual words
//               sent will have words replaced by appropriate UNK tokens
// this lets us use pretrained embeddings, when available, for words that were OOV in the
// srl training data
Expression log_prob_srl(ComputationGraph* hg, const SRL::Sentence& sent, const vector<int>& v, const vector<SRL::Argument>& sr,
                     double *v_right, double *v_pred_base, double *v_gold_base,
		     double *argu_right, double *argu_pred_base, double *argu_gold_base,
		     vector<int> *v_result, vector<SRL::Argument> *sr_result,
		     bool train) {

if(params.debug) cerr<<"sent size: "<<sent.size()<<"\n";

    l2rbuilder.new_graph(*hg);
    r2lbuilder.new_graph(*hg);
    l2rbuilder.start_new_sequence();
    r2lbuilder.start_new_sequence();
    // variables in the computation graph representing the parameters
    Expression lb = parameter(*hg, p_lb);
    Expression w2l = parameter(*hg, p_w2l);
    Expression p2l;
    if (params.use_pos)
      p2l = parameter(*hg, p_p2l);
    Expression t2l;
    if (pretrained.size()>0)
      t2l = parameter(*hg, p_t2l); 
    
    Expression sent_start = parameter(*hg, p_sent_start);
    Expression sent_end = parameter(*hg, p_sent_end);
    
    Expression input2v = parameter(*hg, p_input2v);
    Expression vbias = parameter(*hg, p_vbias);

    Expression head2argu = parameter(*hg, p_head2argu);
    Expression dep2argu = parameter(*hg, p_dep2argu);
    Expression argubias = parameter(*hg, p_argubias);

    vector<Expression> input_expr;

    for (unsigned i = 0; i < sent.size(); ++i) {
      int wordid = sent.raw[i];
      if (train && (int)singletons.size() > wordid && singletons[wordid] && rand01() > params.unk_prob)
          wordid = sent.unk[i];
      if (!train)
          wordid = sent.unk[i];

if(params.debug) cerr<<termdict.convert(wordid)<<" "<<posdict.convert(sent.pos[i]) << " " << termdict.convert(sent.lc[i])<<"\n";

      Expression w =lookup(*hg, p_w, wordid);
      if(train) w = dropout(w,params.pdrop);
      vector<Expression> args = {lb, w2l, w}; // learn embeddings
      if (params.use_pos) { // learn POS tag?
        Expression p = lookup(*hg, p_p, sent.pos[i]);
        if(train) p = dropout(p, params.pdrop);
        args.push_back(p2l);
        args.push_back(p);
      }
      if (pretrained.size() > 0 &&  pretrained.count(sent.lc[i])) {  // include fixed pretrained vectors?
        Expression t = const_lookup(*hg, p_t, sent.lc[i]);
        if(train) t = dropout(t, params.pdrop);
        args.push_back(t2l);
        args.push_back(t);
      }
      else{
        args.push_back(t2l);
        args.push_back(zeroes(*hg,{params.pretrained_dim}));
      }
      input_expr.push_back(rectify(affine_transform(args)));
    }
if(params.debug)	std::cerr<<"lookup table ok\n";
    vector<Expression> l2r(sent.size());
    vector<Expression> r2l(sent.size());
//    Expression l2r_s = l2rbuilder.add_input(sent_start);
//    Expression r2l_e = r2lbuilder.add_input(sent_end);
    l2rbuilder.add_input(sent_start);
    r2lbuilder.add_input(sent_end);
    for (unsigned i = 0; i < sent.size(); ++i) {
      l2r[i] = l2rbuilder.add_input(input_expr[i]);
      r2l[sent.size() - 1 - i] = r2lbuilder.add_input(input_expr[sent.size()-1-i]);
    }
//    Expression l2r_e = l2rbuilder.add_input(sent_end);
//    Expression r2l_s = r2lbuilder.add_input(sent_start);
    l2rbuilder.add_input(sent_end);
    r2lbuilder.add_input(sent_start);
    vector<Expression> input(sent.size());
    for (unsigned i = 0; i < sent.size(); ++i) {
      input[i] = concatenate({l2r[i],r2l[i]});
    }
if(params.debug)	std::cerr<<"bilstm ok\n";
    vector<Expression> log_probs;
    
    for (unsigned i = 0; i < sent.size(); ++i) {
      //possible sense
      vector<unsigned> possibles;
      if(possible_senses.find(sent.raw[i]) != possible_senses.end()){
          possibles = possible_senses[sent.raw[i]];
      }
      else{
          possibles = all_possible_senses;
      }
      Expression vdiste = log_softmax(affine_transform({vbias, input2v, input[i]}), possibles);
      vector<float> vdist = as_vector(hg->incremental_forward(vdiste));
      int best = possibles[0];
      float best_score = vdist[possibles[0]];
      bool gfind = (best == v[i]);
      for(unsigned j = 1; j < possibles.size(); j ++){
      	if (best_score < vdist[possibles[j]]){
		best_score = vdist[possibles[j]];
		best = possibles[j];
	}
        if(possibles[j] == v[i]) gfind = true;
      }
      cerr<<termdict.convert(sent.raw[i])<<" "<<vdict.convert(v[i])<<"\n";
      cerr<<"possible sense: ";
      for(unsigned j = 0; j < possibles.size(); j ++) cerr<<vdict.convert(possibles[j])<<" ";
	cerr<<"\n";
      assert(gfind);
      if(v_result) v_result->push_back(best);
      log_probs.push_back(pick(vdiste, v[i]));

      if(v[i] == best && vdict.convert(v[i]) != "NONE") if(v_right) *v_right += 1;
      if(vdict.convert(v[i]) != "NONE") if(v_gold_base) *v_gold_base += 1;
      if(vdict.convert(best) != "NONE") if(v_pred_base) *v_pred_base += 1;

    }
if(params.debug)	std::cerr<<"verb loss done\n";
    
    for (unsigned i = 0; i < sent.size(); ++i){
      for (unsigned j = 0; j < sent.size(); ++j){
        if(i == j) continue;
	Expression argudiste = log_softmax(affine_transform({argubias, head2argu, input[i], dep2argu, input[j]}));
	vector<float> argudist = as_vector(hg->incremental_forward(argudiste));
	int best = 0;
        float best_score = argudist[0];
        for(unsigned k = 1; k < argudist.size(); k ++){
          if (best_score < argudist[k]){
                best_score = argudist[k];
                best = k;
          }
        }
	int l = find(i, j, sr);

	if(argudict.convert(l) != "NONE") if(sr_result) sr_result->push_back(SRL::Argument(j, i, l));
        log_probs.push_back(pick(argudiste, l));

	if(l == best && argudict.convert(l) != "NONE") if(argu_right) *argu_right += 1;
	if(argudict.convert(l) != "NONE") if(argu_gold_base) *argu_gold_base += 1;
	if(argudict.convert(best) != "NONE") if(argu_pred_base) *argu_pred_base += 1;
      }         
    }
if(params.debug)	std::cerr<<"argument loss done\n";
    Expression tot_neglogprob = -sum(log_probs);
    assert(tot_neglogprob.pg != nullptr);
    return tot_neglogprob;
  }
};

void signal_callback_handler(int /* signum */) {
  if (requested_stop) {
    cerr << "\nReceived SIGINT again, quitting.\n";
    _exit(1);
  }
  cerr << "\nReceived SIGINT terminating optimization early...\n";
  requested_stop = true;
}

int main(int argc, char** argv) {
  DynetParams dynet_params = extract_dynet_params(argc, argv);
  dynet_params.random_seed = 1989121013;
  dynet::initialize(dynet_params);
  cerr << "COMMAND:"; 
  for (unsigned i = 0; i < static_cast<unsigned>(argc); ++i) cerr << ' ' << argv[i];
  cerr << endl;
  unsigned status_every_i_iterations = 100;

  get_args(argc, argv, params);

  assert(params.unk_prob >= 0.); assert(params.unk_prob <= 1.);
  ostringstream os;
  os << "srl_" << (params.use_pos ? "pos" : "nopos")
     << '_' << params.layers
     << '_' << params.input_dim
     << '_' << params.pos_dim
     << '_' << params.bilstm_input_dim
     << '_' << params.bilstm_hidden_dim
     << "-pid" << getpid() << ".params";

  const string fname = os.str();
  cerr << "Writing parameters to file: " << fname << endl;

//=====================================================================================================================
  {
  std::string word;
  ifstream ifs(params.train_dict.c_str());
  while(ifs>>word) train_dict[word] = 1;
  ifs.close();
  }

  if (params.words_file != "") {
    cerr << "Loading from " << params.words_file << " with" << params.pretrained_dim << " dimensions\n";
    ifstream in(params.words_file.c_str());
    string line;
    getline(in, line);
    vector<float> v(params.pretrained_dim, 0);
    string word;
    while (getline(in, line)) {
      istringstream lin(line);
      lin >> word;
      for (unsigned i = 0; i < params.pretrained_dim; ++i) lin >> v[i];
      unsigned id = termdict.convert(word);
      pretrained[id] = v;
    }
  }

  SRL::SRLOracle corpus(&termdict, &vdict, &posdict, &argudict);
  corpus.load_oracle(params.train_file, train_dict);
  
  termdict.freeze();
  termdict.set_unk("UNK");
  vdict.freeze();
  argudict.convert("NONE");
  argudict.freeze();
  posdict.freeze();
  
  {
    for (unsigned i = 0; i < corpus.size(); i ++){
    	SRL::Sentence& sent = corpus.sents[i];
	std::vector<int>& vs = corpus.vs[i];
	
	assert(sent.size() == vs.size());
	for(unsigned j = 0; j < vs.size(); j ++){
	    if(vdict.convert(vs[j]) != "NONE"){
	    	if(possible_senses.find(sent.raw[j]) == possible_senses.end()){
		    std::vector<unsigned> tmp;
		    tmp.push_back(vs[j]);
		    tmp.push_back(vdict.convert("NONE"));
		    possible_senses[sent.raw[j]] = tmp;
		}
		else{
		    bool find = false;
		    std::vector<unsigned>& tmp = possible_senses[sent.raw[j]];
		    for (unsigned k = 0; k < tmp.size(); k ++){
			if(tmp[k] == vs[j]){
			    find = true;
			    break;
			}
		    }
		    if(find == false) possible_senses[sent.raw[j]].push_back(vs[j]);
		}
	    }
	}
    }
    for (unsigned i = 0; i < vdict.size(); i ++){
        all_possible_senses.push_back(i);
    }
  }

  {  // compute the singletons in the srl's training data
    unordered_map<unsigned, unsigned> counts;
    for (auto& sent : corpus.sents)
      for (auto word : sent.raw) counts[word]++;
    singletons.resize(termdict.size(), false);
    for (auto wc : counts)
      if (wc.second == 1) singletons[wc.first] = true;
  }

  ARG_SIZE = argudict.size();
  POS_SIZE = posdict.size();
  VOCAB_SIZE = termdict.size();
  VERB_SIZE = vdict.size();

  cerr<<"verb:\n";
  for(unsigned i = 0; i < vdict.size(); i ++){
    cerr<<i<<":"<<vdict.convert(i)<<"\n";
  }

  cerr<<"postag:\n";
  for(unsigned i = 0; i < posdict.size(); i ++){
    cerr<<i<<":"<<posdict.convert(i)<<"\n";
  }

  cerr<<"argument:\n";
  for(unsigned i = 0; i < argudict.size(); i ++){
    cerr<<i<<":"<<argudict.convert(i)<<"\n";
  }

  SRL::SRLOracle dev_corpus(&termdict, &vdict, &posdict, &argudict);
  if(params.dev_file != "") dev_corpus.load_oracle(params.dev_file, train_dict);
  SRL::SRLOracle test_corpus(&termdict, &vdict, &posdict, &argudict);
  if(params.test_file != "") dev_corpus.load_oracle(params.test_file, train_dict);


//==========================================================================================================================
  
  Model model;
  SRLBuilder srl(&model, pretrained);
  if (params.model_file != "") {
    TextFileLoader loader(params.model_file);
    loader.populate(model);
  }
  
  //TRAINING
  if (params.train) {
    signal(SIGINT, signal_callback_handler);

    Trainer* sgd = NULL;
    unsigned method = params.train_methods;
    if(method == 0)
        sgd = new SimpleSGDTrainer(model);
    else if(method == 1)
        sgd = new MomentumSGDTrainer(model);
    else if(method == 2){
        sgd = new AdagradTrainer(model);
//        sgd->clipping_enabled = false;
    }
    else if(method == 3){
        sgd = new AdamTrainer(model);
//        sgd->clipping_enabled = false;
    }

    vector<unsigned> order(corpus.sents.size());
    for (unsigned i = 0; i < corpus.sents.size(); ++i)
      order[i] = i;
    double tot_seen = 0;
    status_every_i_iterations = min(status_every_i_iterations, corpus.size());
    unsigned si = corpus.size();
    cerr << "NUMBER OF TRAINING SENTENCES: " << corpus.size() << endl;

    double best_v_f = 0;
    double best_argu_f = 0;
    bool first = true;
    int iter = -1;
    while(!requested_stop) {
      ++iter;
      double words = 0;

      double v_right = 0;
      double v_pred_base = 0;
      double v_gold_base = 0;

      double argu_right = 0;
      double argu_pred_base = 0;
      double argu_gold_base = 0;

      double llh = 0;

      auto time_start = chrono::system_clock::now();
      for (unsigned sii = 0; sii < status_every_i_iterations; ++sii) {
           if (si == corpus.size()) {
             si = 0;
             if (first) { first = false; } else { sgd->update_epoch(); }
             cerr << "**SHUFFLE\n";
             random_shuffle(order.begin(), order.end());
           }
           tot_seen += 1;
   	   auto& sentence = corpus.sents[order[si]];
           const vector<int>& v = corpus.vs[order[si]];
	   const vector<SRL::Argument>& sr = corpus.srs[order[si]];

	   ComputationGraph hg;
           Expression nll = srl.log_prob_srl(&hg, sentence, v, sr,
						&v_right, &v_pred_base, &v_gold_base,
						&argu_right, &argu_pred_base, &argu_gold_base,
						NULL, NULL,
						true);

           double lp = as_scalar(hg.incremental_forward(nll));
           if (lp < 0) {
             cerr << "Log prob < 0 on sentence " << order[si] << ": lp=" << lp << endl;
             assert(lp >= 0.0);
           }
           hg.backward(nll);
           sgd->update();
           llh += lp;
           ++si;
	   words += sentence.size();
      }
      sgd->status(); 
      auto time_now = chrono::system_clock::now();
      auto dur = chrono::duration_cast<chrono::milliseconds>(time_now - time_start);

      double v_f = 0;
      if(v_pred_base !=0) v_f = 2*v_right / (v_pred_base + v_gold_base);
      double argu_f = 0;
      if(argu_pred_base != 0) argu_f = 2*argu_right / (argu_pred_base + argu_gold_base);
 
      cerr << "update #" << iter << " (epoch " << (tot_seen / corpus.sents.size()) <<")"
           << " per-input-ppl: " << exp(llh / words)
           << " per-sent-ppl: " << exp(llh / status_every_i_iterations)
           << " verb-f1: " << v_f
	   << " argument-f1: " << argu_f
           << " [" << dur.count() / (double)status_every_i_iterations << "ms per instance]" << endl;

      static int logc = 0;
      ++logc;

      if (logc % 25 == 1) { // report on dev set
        unsigned dev_size = dev_corpus.size();
        double words = 0;

        double v_right = 0;
        double v_pred_base = 0;
        double v_gold_base = 0;

        double argu_right = 0;
        double argu_pred_base = 0;
        double argu_gold_base = 0;

        double llh = 0;

        auto time_start = chrono::system_clock::now();
        for (unsigned sii = 0; sii < dev_size; ++sii) {

	   auto& sentence = dev_corpus.sents[sii];
           const vector<int>& v = dev_corpus.vs[sii];
           const vector<SRL::Argument>& sr = dev_corpus.srs[sii];

           ComputationGraph hg;
           Expression nll = srl.log_prob_srl(&hg, sentence, v, sr,
           					&v_right, &v_pred_base, &v_gold_base,
                    				&argu_right, &argu_pred_base, &argu_gold_base,
                                                NULL, NULL,
                                                false);
           double lp = as_scalar(hg.incremental_forward(nll));
           llh += lp;
	   words += sentence.size();
	}
	auto time_end = chrono::system_clock::now();
        auto dur = chrono::duration_cast<chrono::milliseconds>(time_end - time_start);

	double v_f = 0;
      	if(v_pred_base !=0) v_f = 2*v_right / (v_pred_base + v_gold_base);
      	double argu_f = 0;
      	if(argu_pred_base != 0) argu_f = 2*argu_right / (argu_pred_base + argu_gold_base);
        cerr << "  **dev (iter=" << iter << " epoch=" << (tot_seen / corpus.size()) << ")\t"
                <<" per-input-ppl: " << exp(llh / words)
                <<" per-sent-ppl: " << exp(llh / dev_size)
		<<" verb-f1: " << v_f
                <<" argument-f1: " << argu_f
                <<"\t[" << dev_size << " sents in " << dur.count() << " ms]" << endl;

        if (v_f > best_v_f) {
	  best_v_f = v_f;
	  ostringstream part_os;
          part_os << "predicate"
                << '_' << params.layers
                << '_' << params.input_dim
                << '_' << params.pos_dim
                << '_' << params.bilstm_input_dim
                << '_' << params.bilstm_hidden_dim
                << "-pid" << getpid()
                << "-part" << (tot_seen/corpus.size()) << ".params";
          const string part = part_os.str();

	  TextFileSaver saver("model/"+part);
          saver.save(model);
        }
 
        if (argu_f > argu_f) {
	  best_argu_f = argu_f;
      	  ostringstream part_os;
	  part_os << "argument"
     		<< '_' << params.layers
     		<< '_' << params.input_dim
     		<< '_' << params.pos_dim
     		<< '_' << params.bilstm_input_dim
     		<< '_' << params.bilstm_hidden_dim
     		<< "-pid" << getpid()
		<< "-part" << (tot_seen/corpus.size()) << ".params";
	  const string part = part_os.str();
	  TextFileSaver saver("model/"+part);
	  saver.save(model);  
        }
      }
    }
    delete sgd;
  } // should do training?
  else{ // do test evaluation
        ofstream out("test.out");
        unsigned test_size = test_corpus.size();
	double words = 0;

      	double v_right = 0;
      	double v_pred_base = 0;
      	double v_gold_base = 0;

      	double argu_right = 0;
      	double argu_pred_base = 0;
      	double argu_gold_base = 0;

      	double llh = 0;

        auto time_start = chrono::system_clock::now();
	for (unsigned sii = 0; sii < test_size; ++sii) {
        	auto& sentence = test_corpus.sents[sii];
           	const vector<int>& v = test_corpus.vs[sii];
           	const vector<SRL::Argument>& sr = test_corpus.srs[sii];

           	ComputationGraph hg;
           	Expression nll = srl.log_prob_srl(&hg, sentence, v, sr,
                                                &v_right, &v_pred_base, &v_gold_base,
                                                &argu_right, &argu_pred_base, &argu_gold_base,
                                                NULL, NULL,
                                                true);

           	double lp = as_scalar(hg.incremental_forward(nll));
           	llh += lp;
           	words += sentence.size();

	}
	auto time_end = chrono::system_clock::now();
        auto dur = chrono::duration_cast<chrono::milliseconds>(time_end - time_start);

        double v_f = 0;
        if(v_pred_base !=0) v_f = 2*v_right / (v_pred_base + v_gold_base);
        double argu_f = 0;
        if(argu_pred_base != 0) argu_f = 2*argu_right / (argu_pred_base + argu_gold_base);
	cerr << "  TEST llh= " << llh
                <<" per-input-ppl: " << exp(llh / words)
                <<" per-sent-ppl: " << exp(llh / test_size)
                <<" verb-f1: " << v_f
                <<" argument-f1: " << argu_f
		<<"\t[" << test_size << " sents in " << dur.count() << " ms]" << endl;
  }
}

