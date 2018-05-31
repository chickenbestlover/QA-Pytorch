import time
import argparse
import torch
import msgpack
from drqa.model_ELMo import DocReaderModel
from drqa.utils import str2bool
from prepro_ELMo import annotate, to_id, init
import urllib.request
import time
from allennlp.modules.elmo import Elmo, batch_to_ids
from termcolor import colored
import random
from sys import exit

"""
This script serves as a template to be modified to suit all possible testing environments, including and not limited 
to files (json, xml, csv, ...), web service, databases and so on.
To change this script to batch model, simply modify line 70 from "BatchGen([model_in], batch_size=1, ...)" to 
"BatchGen([model_in_1, model_in_2, ...], batch_size=batch_size, ...)".
"""

class SampleGen:
    def __init__(self, gpu, pos_size, ner_size):
        """
        input:
            data - list of lists
            batch_size - int
        """
        self.gpu = gpu
        self.pos_size = pos_size
        self.ner_size = ner_size

        options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
        self.elmo = Elmo(options_file, weight_file, 2, dropout=0).to('cuda' if self.gpu else 'cpu')

    def process(self,batch):
            batch_size = len(batch)
            batch = list(zip(*batch))
            assert len(batch) == 10

            context_len = max(len(x) for x in batch[1])
            context_id = torch.LongTensor(batch_size, context_len).fill_(0)
            for i, doc in enumerate(batch[1]):
                context_id[i, :len(doc)] = torch.LongTensor(doc)

            feature_len = len(batch[2][0][0])

            context_feature = torch.Tensor(batch_size, context_len, feature_len).fill_(0)
            for i, doc in enumerate(batch[2]):
                for j, feature in enumerate(doc):
                    context_feature[i, j, :] = torch.Tensor(feature)

            context_tag = torch.Tensor(batch_size, context_len, self.pos_size).fill_(0)
            for i, doc in enumerate(batch[3]):
                for j, tag in enumerate(doc):
                    context_tag[i, j, tag] = 1

            context_ent = torch.Tensor(batch_size, context_len, self.ner_size).fill_(0)
            for i, doc in enumerate(batch[4]):
                for j, ent in enumerate(doc):
                    context_ent[i, j, ent] = 1

            question_len = max(len(x) for x in batch[5])
            question_id = torch.LongTensor(batch_size, question_len).fill_(0)
            for i, doc in enumerate(batch[5]):
                question_id[i, :len(doc)] = torch.LongTensor(doc)

            context_mask = torch.eq(context_id, 0)
            question_mask = torch.eq(question_id, 0)
            text = list(batch[6])
            span = list(batch[7])

            context_text = list(batch[-2])
            question_text = list(batch[-1])
            context_elmo = batch_to_ids(context_text)
            question_elmo = batch_to_ids(question_text)

            if self.gpu:
                context_id = context_id.pin_memory()
                context_feature = context_feature.pin_memory()
                context_tag = context_tag.pin_memory()
                context_ent = context_ent.pin_memory()
                context_mask = context_mask.pin_memory()
                question_id = question_id.pin_memory()
                question_mask = question_mask.pin_memory()
                context_elmo = context_elmo.pin_memory().cuda()
                question_elmo = question_elmo.pin_memory().cuda()

            context_elmo = self.elmo(context_elmo)['elmo_representations']
            question_elmo = self.elmo(question_elmo)['elmo_representations']

            return (context_id, context_feature, context_tag, context_ent, context_mask,
                       question_id, question_mask, text, span, context_elmo, question_elmo)


def translate(text,target='en',verbose=True):
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








client_id = "oEooJH6eT7xUFD5CT95j"
client_secret = "KPrgMruWXq"

#client_id = "ox4piK3YeFv28nLO89dN"
#client_secret = "LOUsauFrbP"

parser = argparse.ArgumentParser(
    description='Interact with document reader model.'
)
parser.add_argument('--model-file', default='models/best_model.pt',
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
with open('SQuAD/meta.msgpack', 'rb') as f:
    meta = msgpack.load(f, encoding='utf8')
embedding = torch.Tensor(meta['embedding'])
opt['pretrained_words'] = True
opt['vocab_size'] = embedding.size(0)
opt['embedding_dim'] = embedding.size(1)
opt['pos_size'] = len(meta['vocab_tag'])
opt['ner_size'] = len(meta['vocab_ent'])
opt['cuda'] = args.cuda
model = DocReaderModel(opt, embedding, state_dict)
w2id = {w: i for i, w in enumerate(meta['vocab'])}
tag2id = {w: i for i, w in enumerate(meta['vocab_tag'])}
ent2id = {w: i for i, w in enumerate(meta['vocab_ent'])}
init()
inputGen = SampleGen(gpu=args.cuda,pos_size=opt['pos_size'],ner_size =opt['ner_size'])
print(opt)
print('[Data loaded.] \n')

while True:
    id_ = 0
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

    id_ += 1
    start_time = time.time()

    try:
        annotated = annotate(('interact-{}'.format(id_), evidence, question), meta['wv_cased'])
        model_in = to_id(annotated, w2id, tag2id, ent2id)
        model_in = model_in + tuple([annotated[1], annotated[5]])
        model_in = inputGen.process([model_in])
        prediction = model.predict(model_in)[0]
        rescode_p, prediction, _ = translate(prediction,lang_q,verbose=False)
    except:
        print(colored('Error  : ', 'red'), 'Please try again with another natural language input. \n')
        continue
    end_time = time.time()
    print(colored('Time    : ','green'),'{:.4f}s'.format(end_time - start_time))
    print(colored('Answer  : ','green'),'{}'.format(prediction),'\n')




