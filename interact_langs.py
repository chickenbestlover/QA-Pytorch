import time
import argparse
import torch
import msgpack
from drqa.model import DocReaderModel
from drqa.utils import str2bool
from prepro import annotate, to_id, init
from train import BatchGen
import os
import sys
import urllib.request

"""
This script serves as a template to be modified to suit all possible testing environments, including and not limited 
to files (json, xml, csv, ...), web service, databases and so on.
To change this script to batch model, simply modify line 70 from "BatchGen([model_in], batch_size=1, ...)" to 
"BatchGen([model_in_1, model_in_2, ...], batch_size=batch_size, ...)".
"""

#client_id = "oEooJH6eT7xUFD5CT95j"
#client_secret = "KPrgMruWXq"

client_id = "ox4piK3YeFv28nLO89dN"
client_secret = "LOUsauFrbP"

parser = argparse.ArgumentParser(
    description='Interact with document reader model.'
)
parser.add_argument('--model-file', default='models/best_model.pt',
                    help='path to model file')
parser.add_argument("--cuda", type=str2bool, nargs='?',
                    const=True, default=torch.cuda.is_available(),
                    help='whether to use GPU acceleration.')
args = parser.parse_args()


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
BatchGen.pos_size = opt['pos_size']
BatchGen.ner_size = opt['ner_size']
model = DocReaderModel(opt, embedding, state_dict)
w2id = {w: i for i, w in enumerate(meta['vocab'])}
tag2id = {w: i for i, w in enumerate(meta['vocab_tag'])}
ent2id = {w: i for i, w in enumerate(meta['vocab_ent'])}
init()

def translate(text,target='en'):
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
        print(lang)
        if lang ==target or lang == 'unk':
            pass
        else:
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
        print("Error Code:" + rescode)
        return rescode, text, lang

    return rescode, text, lang

while True:
    id_ = 0
    try:
        while True:
            evidence = input('Evidence: ')
            if evidence.strip():
                break
        while True:
            question = input('Question: ')
            if question.strip():
                break
    except EOFError:
        print()
        break
    id_ += 1
    start_time = time.time()

    rescode_e, evidence, lang_e = translate(evidence,'en')
    rescode_q, question, lang_q = translate(question,'en')


    annotated = annotate(('interact-{}'.format(id_), evidence, question), meta['wv_cased'])
    model_in = to_id(annotated, w2id, tag2id, ent2id)
    model_in = next(iter(BatchGen([model_in], batch_size=1, gpu=args.cuda, evaluation=True)))
    prediction = model.predict(model_in)[0]


    rescode_p, prediction, _ = translate(prediction,lang_q)

    end_time = time.time()
    print('Answer: {}'.format(prediction))
    print('Time: {:.4f}s'.format(end_time - start_time))




