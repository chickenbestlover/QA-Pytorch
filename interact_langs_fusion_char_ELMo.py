
import argparse
import torch
from drqa.model_fusion_char_elmo import DocReaderModel
from drqa.utils import str2bool
import urllib.request
import time
from allennlp.modules.elmo import Elmo, batch_to_ids
from termcolor import colored
from sys import exit
import os
import numpy as np
import ujson as json
from prepro_fusion_ELMo import word_tokenize, convert_idx, pre_proc
from collections import Counter




class Inference(object):

    def __init__(self, opt, device='cuda'):
        self.char_limit = 16
        self.para_limit = 1000
        self.ques_limit = 100
        self.device = device
        self.opt=opt
        self.word2idx_dict = self.get_data(args.data_path + 'word2id.json')
        self.char2idx_dict = self.get_data(args.data_path + 'char2id.json')
        self.pos2idx_dict = self.get_data(args.data_path + 'pos2id.json')
        self.ner2idx_dict = self.get_data(args.data_path + 'ner2id.json')
        options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/" \
                       "2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/" \
                      "2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
        self.elmo = Elmo(options_file, weight_file, 2, dropout=0).to(device)

        self.model = DocReaderModel(opt, embedding=None, state_dict=state_dict)

    def get_data(self,filename):
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def compute_mask(self, x):
        return torch.eq(x, 0).to(self.device)

    def convert_tokens(self, context, spans, y1, y2) :
        start_idx = spans[y1.item()][0]
        end_idx = spans[y2.item()][1]
        answer = context[start_idx : end_idx]
        return answer

    def response(self, context, question):
        preprocessed, spans = self.prepro(context,question)
        logit1, logit2 =self.model.network.forward(*preprocessed)
        y1,y2 = self.model.get_predictions(logit1,logit2)
        context = context.replace("''", '" ').replace("``", '" ')
        context = pre_proc(context)
        answer = self.convert_tokens(context,spans,y1,y2)
        return answer

    def prepro(self, context, question):
        context = context.replace("''", '" ').replace("``", '" ')
        context = pre_proc(context)
        context_tokens, context_tags, context_ents, context_lemmas = word_tokenize(context) #text, tag, ent, lemma
        context_lower_tokens = [w.lower() for w in context_tokens]
        context_chars = [list(token) for token in context_tokens]
        spans = convert_idx(context, context_tokens)

        counter_ = Counter(context_lower_tokens)
        tf_total = len(context_lower_tokens)
        context_tf = [float(counter_[w]) / float(tf_total) for w in context_lower_tokens]

        ques = question.replace("''", '" ').replace("``", '" ')
        ques = pre_proc(ques)
        ques_tokens, ques_tags, ques_ents, ques_lemmas = word_tokenize(ques)
        ques_lower_tokens = [w.lower() for w in ques_tokens]
        ques_chars = [list(token) for token in ques_tokens]

        ques_lemma = {lemma if lemma != '-PRON-' else lower for lemma, lower in zip(ques_lemmas, ques_lower_tokens)}

        ques_tokens_set = set(ques_tokens)
        ques_lower_tokens_set = set(ques_lower_tokens)
        match_origin = [w in ques_tokens_set for w in context_tokens]
        match_lower = [w in ques_lower_tokens_set for w in context_lower_tokens]
        match_lemma = [(c_lemma if c_lemma != '-PRON-' else c_lower) in ques_lemma for (c_lemma, c_lower) in
                       zip(context_lemmas, context_lower_tokens)]


        example = {"context_tokens": context_tokens,
                   "context_chars": context_chars,
                   "match_origin": match_origin,
                   "match_lower": match_lower,
                   "match_lemma": match_lemma,
                   "context_pos": context_tags,
                   "context_ner": context_ents,
                   "context_tf": context_tf,
                   "ques_tokens": ques_tokens,
                   "ques_pos": ques_tags,
                   "ques_ner": ques_ents,
                   "ques_chars": ques_chars}



        context_idxs = np.zeros([self.para_limit], dtype=np.int32)
        context_elmo_tokens = example['context_tokens']

        match_origin = np.zeros([self.para_limit], dtype=np.int32)
        match_lower = np.zeros([self.para_limit], dtype=np.int32)
        match_lemma = np.zeros([self.para_limit], dtype=np.int32)
        context_tf = np.zeros([self.para_limit], dtype = np.float32)
        context_pos_idxs = np.zeros([self.para_limit], dtype=np.int32)
        context_ner_idxs = np.zeros([self.para_limit], dtype=np.int32)
        context_char_idxs = np.zeros([self.para_limit, self.char_limit], dtype=np.int32)

        ques_idxs = np.zeros([self.ques_limit], dtype=np.int32)
        ques_elmo_tokens = example['ques_tokens']
        ques_pos_idxs = np.zeros([self.ques_limit], dtype=np.int32)
        ques_ner_idxs = np.zeros([self.ques_limit], dtype=np.int32)
        ques_char_idxs = np.zeros([self.ques_limit, self.char_limit], dtype=np.int32)


        def _get_word(word):
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in self.word2idx_dict:
                    return self.word2idx_dict[each]
            return 1
        def _get_pos(pos) :
            if pos in self.pos2idx_dict :
                return self.pos2idx_dict[pos]
            return 1
        def _get_ner(ner) :
            if ner in self.ner2idx_dict :
                return self.ner2idx_dict[ner]
            return 1

        def _get_char(char):
            if char in self.char2idx_dict:
                return self.char2idx_dict[char]
            return 1

        for i, token in enumerate(example["context_tokens"]):
            context_idxs[i] = _get_word(token)
        for i, match in enumerate(example["match_origin"]) :
            match_origin[i] = 1 if match == True else 0
        for i, match in enumerate(example["match_lower"]) :
            match_lower[i] = 1 if match == True else 0
        for i, match in enumerate(example["match_lemma"]) :
            match_lemma[i] = 1 if match == True else 0

        for i, tf in enumerate(example['context_tf']) :
            context_tf[i] = tf

        for i, pos in enumerate(example['context_pos']) :
            context_pos_idxs[i] = _get_pos(pos)
        for i, ner in enumerate(example['context_ner']) :
            context_ner_idxs[i] = _get_ner(ner)

        for i, token in enumerate(example["ques_tokens"]):
            ques_idxs[i] = _get_word(token)

        for i, pos in enumerate(example['ques_pos']) :
            ques_pos_idxs[i] = _get_pos(pos)
        for i, ner in enumerate(example['ques_ner']) :
            ques_ner_idxs[i] = _get_ner(ner)


        for i, token in enumerate(example["context_chars"]):
            for j, char in enumerate(token):
                if j == self.char_limit:
                    break
                context_char_idxs[i, j] = _get_char(char)

        for i, token in enumerate(example["ques_chars"]):
            for j, char in enumerate(token):
                if j == self.char_limit:
                    break
                ques_char_idxs[i, j] = _get_char(char)


        passage_ids = torch.LongTensor([context_idxs.tolist()]).to(self.device)
        passage_char_ids = torch.LongTensor([context_char_idxs.tolist()]).to(self.device)
        passage_pos_ids = torch.LongTensor([context_pos_idxs.tolist()]).to(self.device)
        passage_ner_ids = torch.LongTensor([context_ner_idxs.tolist()]).to(self.device)
        passage_match_origin = torch.FloatTensor([match_origin.tolist()]).to(self.device)
        passage_match_lower = torch.FloatTensor([match_lower.tolist()]).to(self.device)
        passage_match_lemma = torch.FloatTensor([match_lemma.tolist()]).to(self.device)
        passage_tf = torch.FloatTensor([context_tf.tolist()]).to(self.device)

        ques_ids = torch.LongTensor([ques_idxs.tolist()]).to(self.device)
        ques_char_ids = torch.LongTensor([ques_char_idxs.tolist()]).to(self.device)
        ques_pos_ids = torch.LongTensor([ques_pos_idxs.tolist()]).to(self.device)
        ques_ner_ids = torch.LongTensor([ques_ner_idxs.tolist()]).to(self.device)


        passage_elmo_ids = batch_to_ids([context_elmo_tokens]).to(self.device)
        question_elmo_ids = batch_to_ids([ques_elmo_tokens]).to(self.device)

        p_lengths = passage_ids.ne(0).long().sum(1)
        q_lengths = ques_ids.ne(0).long().sum(1)

        passage_maxlen = int(torch.max(p_lengths, 0)[0])
        ques_maxlen = int(torch.max(q_lengths, 0)[0])

        passage_ids = passage_ids[:, :passage_maxlen]
        passage_char_ids = passage_char_ids[:, :passage_maxlen]
        passage_pos_ids = passage_pos_ids[:, :passage_maxlen]
        passage_ner_ids = passage_ner_ids[:, :passage_maxlen]
        passage_match_origin = passage_match_origin[:, :passage_maxlen]
        passage_match_lower = passage_match_lower[:, :passage_maxlen]
        passage_match_lemma = passage_match_lemma[:, :passage_maxlen]
        passage_tf = passage_tf[:, :passage_maxlen]
        ques_ids = ques_ids[:, :ques_maxlen]
        ques_char_ids = ques_char_ids[:, :ques_maxlen]
        ques_pos_ids = ques_pos_ids[:, :ques_maxlen]
        ques_ner_ids = ques_ner_ids[:, :ques_maxlen]

        p_mask = self.compute_mask(passage_ids)
        q_mask = self.compute_mask(ques_ids)
        passage_elmo = self.elmo(passage_elmo_ids)['elmo_representations'][0]
        question_elmo = self.elmo(question_elmo_ids)['elmo_representations'][0]

        return (passage_ids,
               passage_char_ids,
               passage_pos_ids,
               passage_ner_ids,
               passage_match_origin.unsqueeze(2).float(),
               passage_match_lower.unsqueeze(2).float(),
               passage_match_lemma.unsqueeze(2).float(),
               passage_tf.unsqueeze(2),
               p_mask,
               ques_ids,
               ques_char_ids,
               ques_pos_ids,
               ques_ner_ids,
               q_mask,
               passage_elmo,
               question_elmo), spans


def translate(text,target='en',verbose=True):
    client_id = "oEooJH6eT7xUFD5CT95j"
    client_secret = "KPrgMruWXq"
    # client_id = "ox4piK3YeFv28nLO89dN"
    # client_secret = "LOUsauFrbP"
    text_enc = urllib.parse.quote(text)
    data = "query=" + text_enc
    url = "https://openapi.naver.com/v1/papago/detectLangs"
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id", client_id)
    request.add_header("X-Naver-Client-Secret", client_secret)
    response = urllib.request.urlopen(request, data=data.encode('utf-8'))
    rescode = response.getcode()
    lang = 'unk'
    if (rescode == 200):
        response_body = response.read()
        lang = response_body.decode('utf-8').split('"')[3]
        if verbose: print(colored('Detected: ','green'), lang,'\n')
        if lang !=target and lang in ('ko,en,ja,zh-CN,zh-TW'):
            data = "source="+lang+"&target="+target+"&text=" + text_enc
            url = "https://openapi.naver.com/v1/papago/n2mt"
            request = urllib.request.Request(url)
            request.add_header("X-Naver-Client-Id", client_id)
            request.add_header("X-Naver-Client-Secret", client_secret)
            response = urllib.request.urlopen(request, data=data.encode('utf-8'))
            rescode = response.getcode()
            if (rescode == 200):
                response_body = response.read()
                text =response_body.decode().split('"')[-2]
            else:
                print("Error Code:" + rescode)
                return rescode, text, lang
        else:
            pass
    else:
        print("Error Code:" + rescode)
        return rescode, text, lang

    return rescode, text, lang





parser = argparse.ArgumentParser(
    description='inferene model.'
)
parser.add_argument('--data_path', default='./SQuAD_fusion/')
parser.add_argument('--model_file', default='models/fusion_charTrue_coveFalse_elmoTrue.pt',
                    help='path to model file')
parser.add_argument("--cuda", type=str2bool, nargs='?',
                    const=True, default=torch.cuda.is_available(),
                    help='whether to use GPU acceleration.')
args = parser.parse_args()

print('[Program starts. Loading data...]')
if args.cuda:
    checkpoint = torch.load(args.model_file)
else:
    checkpoint = torch.load(args.model_file, map_location=lambda storage, loc: storage)
state_dict = checkpoint['state_dict']
opt = checkpoint['config']
opt['vocab_size']=90968
opt['embedding_dim']=300
opt['cuda']=args.cuda
print(opt)
print('[Data loaded.] \n')

context = "In meteorology, precipitation is any product of the condensation " \
          "of atmospheric water vapor that falls under gravity. The main forms " \
          "of precipitation include drizzle, rain, sleet, snow, graupel and hail." \
          "Precipitation forms as smaller droplets coalesce via collision with other " \
          "rain drops or ice crystals within a cloud. Short, intense periods of rain " \
          "in scattered locations are called “showers”."
question = "What causes precipitation to fall?"

infer = Inference(opt, device='cuda' if args.cuda else 'cpu')

while True:

    try:
        try:
            evidence = input(colored('Evidence: ','blue'))
            if evidence.strip(): pass
        except UnicodeDecodeError as e:
            print(e)
            print(colored('Error  : ','red'),'Please try again with another natural language input.\n')
            continue
        try:
            rescode_e, evidence, lang_e = translate(evidence, 'en')
        except:
            pass
        try:
            question = input(colored('Question: ','blue'))
            if question.strip(): pass

        except UnicodeDecodeError as e:
            print(e)
            print(colored('Error  : ', 'red'), 'Please try again with another natural language input. \n')
            continue
        try:
            rescode_q, question, lang_q = translate(question, 'en')
        except:
            lang_q = 'en'
            pass


    except EOFError:
        print()
        break
    except KeyboardInterrupt:
        print("Exit the program.")
        exit()

    start_time = time.time()

    try:
        prediction=infer.response(evidence,question)
        rescode_p, prediction, _ = translate(prediction,lang_q,verbose=False)
    except:
        print(colored('Error  : ', 'red'), 'Please try again with another natural language input. \n')
        continue
    end_time = time.time()
    print(colored('Time    : ','green'),'{:.4f}s'.format(end_time - start_time))
    print(colored('Answer  : ','green'),'{}'.format(prediction),'\n')




