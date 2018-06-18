# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
# Modification:
#   - add 'pos' and 'ner' features.
#   - use gradient hook (instead of tensor copying) for gradient masking
# Origin: https://github.com/facebookresearch/ParlAI/tree/master/parlai/agents/drqa

import torch
import torch.nn as nn
from qa.cove.cove import MTLSTM
from qa.layers import StackedLSTM, Dropout, Summ, PointerNet, SRU
from qa.layers import FullAttention
from qa.layers import WordAttention_multiHead as WordAttention
from qa.layers import FullAttention_multiHead
import pickle as pkl
from allennlp.modules.elmo import Elmo

class ReaderNet(nn.Module):
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN, 'sru': SRU}
    def __init__(self, opt, embedding):
        super(ReaderNet, self).__init__()
        # Store config
        self.opt = opt

        self.device = 'cuda' if opt['cuda'] else 'cpu'

        # Word embeddings
        if embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding, freeze=False)
        else:
            self.embedding = nn.Embedding(opt['vocab_size'],
                                          opt['embedding_dim'],
                                          padding_idx=0)
        if opt['fix_embeddings']:
            self.embedding.weight.requires_grad = False
        else :
            with open(opt['data_path'] + 'tune_word_idx.pkl', 'rb') as f :
                tune_idx = pkl.load(f)
            self.fixed_idx = list(set([i for i in range(self.embedding.weight.shape[0])]) - set(tune_idx))
            self.embedding.weight.data[self.fixed_idx].requires_grad=False
            def embedding_hook(grad, idx = self.fixed_idx):
                grad[idx] = 0
                return grad

            self.embedding.weight.register_hook(embedding_hook)

            fixed_embedding = self.embedding.weight.data[self.fixed_idx].to(self.device)
            fixed_embedding.requires_grad=False
            #self.register_buffer('fixed_embedding', fixed_embedding)
            self.fixed_embedding = fixed_embedding

        if self.opt['use_elmo']:
            weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/" \
                          "2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
            options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/" \
                           "2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
            self.elmo = Elmo(options_file, weight_file, 3, dropout=0)


        self.pos_embeddings = nn.Embedding(opt['pos_size'], opt['pos_dim'], padding_idx=0)

        self.ner_embeddings = nn.Embedding(opt['ner_size'], opt['ner_dim'], padding_idx=0)



        # Input size to RNN: word emb + question emb + manual features
        opt['cove_dim']=600
        opt['elmo_dim']=1024
        opt['embedding_dim']=self.embedding.weight.shape[1] # 300
        doc_input_size = 2 * opt['embedding_dim'] + opt['num_features'] + opt['pos_dim'] + opt['ner_dim']
        question_input_size = opt['embedding_dim'] + opt['pos_dim'] + opt['ner_dim']


        if self.opt['use_char']:
            self.char_embeddings = nn.Embedding(opt['char_size'], opt['char_dim'], padding_idx=0)
            self.char_rnn = self.RNN_TYPES[opt['rnn_type']](input_size = opt['char_dim'],
                                                            hidden_size = opt['char_hidden_size'],
                                                            batch_first = True,
                                                            bidirectional = True,
                                                            num_layers = 1,
                                                            dropout = 0)
            doc_input_size += 2 * opt['char_hidden_size']
            question_input_size += 2 * opt['char_hidden_size']
        if self.opt['use_cove']:
            self.cove_rnn = MTLSTM(opt, embedding=embedding)
            doc_input_size += opt['cove_dim']
            question_input_size += opt['cove_dim']
        if self.opt['use_elmo']:
            doc_input_size += opt['elmo_dim']
            question_input_size += opt['elmo_dim']

        self.word_attention_layer = WordAttention(input_size = opt['embedding_dim'],
                                                  hidden_size = opt['attention_size'],
                                                  dropout = opt['dropout_emb'],
                                                  device= self.device)

        self.low_doc_rnn = StackedLSTM(input_size=doc_input_size,
                                       hidden_size=opt['hidden_size'],
                                       num_layers=opt['num_layers'],
                                       dropout=opt['dropout'],
                                       dropout_rnn=opt['dropout_rnn'],
                                       device= self.device,
                                       rnn_type=self.RNN_TYPES[opt['rnn_type']],
                                       res=opt['use_res'],
                                       norm=opt['use_norm'])

        self.low_ques_rnn = StackedLSTM(input_size=question_input_size,
                                        hidden_size=opt['hidden_size'],
                                        num_layers=opt['num_layers'],
                                        dropout=opt['dropout'],
                                        dropout_rnn=opt['dropout_rnn'],
                                        device= self.device,
                                        rnn_type=self.RNN_TYPES[opt['rnn_type']],
                                        res=opt['use_res'],
                                        norm=opt['use_norm'])

        # Output sizes of low rnn encoders
        high_doc_hidden_size = 2 * opt['hidden_size']
        high_ques_hidden_size = 2 * opt['hidden_size']

        self.high_doc_rnn = StackedLSTM(input_size=high_doc_hidden_size,
                                        hidden_size=opt['hidden_size'],
                                        num_layers=opt['num_layers'],
                                        dropout=opt['dropout'],
                                        dropout_rnn=opt['dropout_rnn'],
                                        device= self.device,
                                        rnn_type=self.RNN_TYPES[opt['rnn_type']],
                                        res=opt['use_res'],
                                        norm=opt['use_norm'])

        self.high_ques_rnn = StackedLSTM(input_size=high_ques_hidden_size,
                                         hidden_size=opt['hidden_size'],
                                         num_layers=opt['num_layers'],
                                         dropout=opt['dropout'],
                                         dropout_rnn=opt['dropout_rnn'],
                                         device= self.device,
                                         rnn_type=self.RNN_TYPES[opt['rnn_type']],
                                         res=opt['use_res'],
                                         norm=opt['use_norm'])

        und_q_word_size = 2 * (2 * opt['hidden_size'])

        self.und_ques_rnn = StackedLSTM(input_size=und_q_word_size,
                                        hidden_size=opt['hidden_size'],
                                        num_layers=opt['num_layers'],
                                        dropout=opt['dropout'],
                                        dropout_rnn=opt['dropout_rnn'],
                                        device= self.device,
                                        rnn_type=self.RNN_TYPES[opt['rnn_type']],
                                        res=opt['use_res'],
                                        norm=opt['use_norm'])

        attention_inp_size = opt['embedding_dim'] + 2 * (2 * opt['hidden_size'])
        if self.opt['use_cove']:
            attention_inp_size += opt['cove_dim']
        if self.opt['use_elmo']:
            attention_inp_size += opt['elmo_dim']

        self.low_attention_layer = FullAttention(input_size = attention_inp_size,
                                                 hidden_size = opt['attention_size'],
                                                 dropout = opt['dropout'],
                                                 device= self.device)

        self.high_attention_layer = FullAttention(input_size=attention_inp_size,
                                                  hidden_size=opt['attention_size'],
                                                  dropout=opt['dropout'],
                                                  device=self.device)

        self.und_attention_layer = FullAttention(input_size=attention_inp_size,
                                                 hidden_size=opt['attention_size'],
                                                 dropout=opt['dropout'],
                                                 device=self.device)

        fuse_inp_size = 5 * (2 * opt['hidden_size'])

        self.fuse_rnn = StackedLSTM(input_size = fuse_inp_size,
                                    hidden_size = opt['hidden_size'],
                                    num_layers = opt['num_layers'],
                                    dropout = opt['dropout'],
                                    dropout_rnn=opt['dropout_rnn'],
                                    device=self.device,
                                    rnn_type=self.RNN_TYPES[opt['rnn_type']],
                                    res=opt['use_res'],
                                    norm=opt['use_norm'])

        self.fuse_attn = FullAttention_multiHead(input_size=2*opt['hidden_size'],
                                                 hidden_size=opt['attention_size'],
                                                 dropout=opt['dropout'],
                                                           n_head=5,
                                                 device=self.device)

        self_attention_inp_size = opt['embedding_dim'] + opt['pos_dim'] + opt['ner_dim']+ \
                                  6 * (2 * opt['hidden_size']) + 1
        if self.opt['use_cove']:
            self_attention_inp_size += opt['cove_dim']
        if self.opt['use_elmo']:
            self_attention_inp_size += opt['elmo_dim']

        self.self_attention_layer = FullAttention(input_size=self_attention_inp_size,
                                                  hidden_size=opt['attention_size'],
                                                  dropout=opt['dropout'],
                                                  device=self.device)

        self.self_rnn = StackedLSTM(input_size = 2 * (2 * opt['hidden_size']),
                                    hidden_size = opt['hidden_size'],
                                    num_layers = opt['num_layers'],
                                    dropout_rnn=opt['dropout_rnn'],
                                    dropout = opt['dropout'],
                                    device=self.device,
                                    rnn_type=self.RNN_TYPES[opt['rnn_type']],
                                    res=opt['use_res'],
                                    norm=opt['use_norm'])
        self.self_attn = FullAttention_multiHead(input_size=2*opt['hidden_size'],
                                                 hidden_size=opt['attention_size'],
                                                 dropout=opt['dropout'],
                                                           n_head=5,
                                                 device=self.device)
        self.summ_layer = Summ(input_size=2 * opt['hidden_size'],
                               dropout=opt['dropout'],
                               device=self.device)

        self.pointer_layer = PointerNet(input_size=2 * opt['hidden_size'],
                                        dropout=opt['dropout'],
                                        device=self.device)


    def reset_parameters(self) :
        if not self.opt['fix_embeddings'] :
            self.embedding.weight.data[self.fixed_idx] = self.fixed_embedding

    def forward(self, x1, x1_char, x1_pos, x1_ner, x1_origin, x1_lower, x1_lemma, x1_tf, x1_mask,
                x2, x2_char, x2_pos, x2_ner, x2_mask, x1_elmo=None, x2_elmo=None):


        ### GloVe ###
        x1_glove_emb = self.embedding(x1)
        x2_glove_emb = self.embedding(x2)

        ### character ###
        if self.opt['use_char'] :
            passage_char_emb = self.char_embeddings(x1_char.contiguous().view(-1, x1_char.size(2)))
            ques_char_emb = self.char_embeddings(x2_char.contiguous().view(-1, x2_char.size(2)))
            d_passage_char_emb = Dropout(passage_char_emb, self.opt['dropout'], self.training, device= self.device)
            d_ques_char_emb = Dropout(ques_char_emb, self.opt['dropout'], self.training, device = self.device)
            _, (h, c) = self.char_rnn(d_passage_char_emb)
            x1_char_states = torch.cat([h[0], h[1]], dim=1).contiguous().view(-1, x1.size(1), 2*self.opt['char_hidden_size'])
            _, (h, c) = self.char_rnn(d_ques_char_emb)
            x2_char_states = torch.cat([h[0], h[1]], dim=1).contiguous().view(-1, x2.size(1), 2*self.opt['char_hidden_size'])

        ### CoVe ###
        if self.opt['use_cove']:
            _, x1_cove_emb = self.cove_rnn(x1, x1_mask)
            _, x2_cove_emb = self.cove_rnn(x2, x2_mask)

        ### ELMo ###
        if self.opt['use_elmo']:
            x1_elmo_embs = self.elmo(x1_elmo)['elmo_representations']
            x2_elmo_embs = self.elmo(x2_elmo)['elmo_representations']

        ### embeddings ###
        x1_pos_emb = self.pos_embeddings(x1_pos)
        x1_ner_emb = self.ner_embeddings(x1_ner)
        x2_pos_emb = self.pos_embeddings(x2_pos)
        x2_ner_emb = self.ner_embeddings(x2_ner)

        ### embedding dropout ###
        x1_glove_emb = Dropout(x1_glove_emb, self.opt['dropout_emb'], self.training, device = self.device)
        x2_glove_emb = Dropout(x2_glove_emb, self.opt['dropout_emb'], self.training, device = self.device)
        if self.opt['use_cove']:
            x1_cove_emb = Dropout(x1_cove_emb, self.opt['dropout_emb'], self.training, device = self.device)
            x2_cove_emb = Dropout(x2_cove_emb, self.opt['dropout_emb'], self.training, device = self.device)
        if self.opt['use_elmo']:
            for i in range(len(x1_elmo_embs)):
                x1_elmo_embs[i] = Dropout(x1_elmo_embs[i], self.opt['dropout_emb'], self.training, device = self.device)
                x2_elmo_embs[i] = Dropout(x2_elmo_embs[i], self.opt['dropout_emb'], self.training, device = self.device)
        word_attention_outputs = self.word_attention_layer.forward(x1_glove_emb, x1_mask, x2_glove_emb, x2_mask)

        x1_word_input = torch.cat(
            [x1_glove_emb, x1_pos_emb, x1_ner_emb, x1_tf, word_attention_outputs, x1_origin, x1_lower, x1_lemma], dim=2)
        x2_word_input = torch.cat([x2_glove_emb, x2_pos_emb, x2_ner_emb], dim=2)

        if self.opt['use_char'] :
            x1_word_input = torch.cat([x1_word_input, x1_char_states], dim=2)
            x2_word_input = torch.cat([x2_word_input, x2_char_states], dim=2)
        if self.opt['use_cove']:
            x1_word_input = torch.cat([x1_word_input, x1_cove_emb], dim=2)
            x2_word_input = torch.cat([x2_word_input, x2_cove_emb], dim=2)
        if self.opt['use_elmo']:
            x1_word_input = torch.cat([x1_word_input, x1_elmo_embs[0]], dim=2)
            x2_word_input = torch.cat([x2_word_input, x2_elmo_embs[0]], dim=2)

        ### low, high, understanding encoding ###
        low_x1_states = self.low_doc_rnn.forward(x1_word_input)
        low_x2_states = self.low_ques_rnn.forward(x2_word_input)

        high_x1_states = self.high_doc_rnn.forward(low_x1_states)
        high_x2_states = self.high_ques_rnn.forward(low_x2_states)

        und_x2_input = torch.cat([low_x2_states, high_x2_states], dim=2)
        und_x2_states = self.und_ques_rnn.forward(und_x2_input)

        ########################################################################################


        ### Full Attention ###

        x1_How = torch.cat([x1_glove_emb, low_x1_states, high_x1_states], dim=2)
        x2_How = torch.cat([x2_glove_emb, low_x2_states, high_x2_states], dim=2)
        if self.opt['use_cove']:
            x1_How = torch.cat([x1_How, x1_cove_emb], dim=2)
            x2_How = torch.cat([x2_How, x2_cove_emb], dim=2)
        if self.opt['use_elmo']:
            x1_How = torch.cat([x1_How, x1_elmo_embs[1]], dim=2)
            x2_How = torch.cat([x2_How, x2_elmo_embs[1]], dim=2)

        low_attention_outputs = self.low_attention_layer.forward(x1_How, x1_mask, x2_How, x2_mask, low_x2_states)
        high_attention_outputs = self.high_attention_layer.forward(x1_How, x1_mask, x2_How, x2_mask, high_x2_states)
        und_attention_outputs = self.und_attention_layer.forward(x1_How, x1_mask, x2_How, x2_mask, und_x2_states)

        fuse_inp = torch.cat([low_x1_states, high_x1_states, low_attention_outputs, high_attention_outputs, und_attention_outputs], dim = 2)

        fused_x1_states = self.fuse_rnn.forward(fuse_inp)
        fused_x1_states = self.fuse_attn.forward(fused_x1_states,x1_mask,und_x2_states,x2_mask,fused_x1_states)

        ### Self Full Attention ###

        x1_How = torch.cat([x1_glove_emb, x1_pos_emb, x1_ner_emb, x1_tf,
                            low_x1_states, high_x1_states,
                            low_attention_outputs, high_attention_outputs,
                            und_attention_outputs, fused_x1_states], dim=2)
        if self.opt['use_cove']:
            x1_How = torch.cat([x1_How, x1_cove_emb], dim=2)
        if self.opt['use_elmo']:
            x1_How = torch.cat([x1_How, x1_elmo_embs[2]], dim=2)

        self_attention_outputs = self.self_attention_layer.forward(x1_How, x1_mask, x1_How, x1_mask, fused_x1_states)

        self_inp = torch.cat([fused_x1_states, self_attention_outputs], dim=2)

        und_x1_states = self.self_rnn.forward(self_inp)
        und_x1_states = self.self_attn.forward(und_x1_states,x1_mask,und_x2_states,x2_mask,und_x1_states)

        ### ques summ vector ###
        init_states = self.summ_layer.forward(und_x2_states, x2_mask)

        ### Pointer Network ###
        logits1, logits2 = self.pointer_layer.forward(und_x1_states, x1_mask, init_states)

        return logits1, logits2


