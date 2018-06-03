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
import torch.nn.functional as F
from drqa.cove.cove import MTLSTM
from drqa.cove.layers import StackedLSTM, Dropout, FullAttention, WordAttention, Summ, PointerNet


class RnnDocReader(nn.Module):
    def __init__(self, opt, embedding):
        super(RnnDocReader, self).__init__()
        # Store config
        self.opt = opt
        self.device = 'cuda' if opt['cuda'] else 'cpu'

        # Word embeddings
        self.embedding = nn.Embedding.from_pretrained(embedding, freeze=False)
        if opt['fix_embeddings']:
            assert opt['tune_partial'] == 0
            self.embedding.weight.requires_grad = False
        elif opt['tune_partial'] > 0:
            assert opt['tune_partial'] + 2 < embedding.size(0)
            offset = self.opt['tune_partial'] + 2

            def embedding_hook(grad, offset=offset):
                grad[offset:] = 0
                return grad

            self.embedding.weight.register_hook(embedding_hook)


        self.pos_embeddings = nn.Embedding(opt['pos_size'], opt['pos_dim'], padding_idx=0)

        self.ner_embeddings = nn.Embedding(opt['ner_size'], opt['ner_dim'], padding_idx=0)



        # Input size to RNN: word emb + question emb + manual features
        opt['cove_dim']=600
        opt['embedding_dim']=self.embedding.weight.shape[1]
        doc_input_size = 2 * opt['embedding_dim'] + opt['num_features'] + opt['cove_dim']+ \
                         opt['pos_dim'] + opt['ner_dim']
        question_input_size = opt['embedding_dim'] + opt['cove_dim'] + opt['pos_dim'] + opt['ner_dim']


        if self.opt['use_char']:
            self.char_embeddings = nn.Embedding(opt['char_size'], opt['char_dim'], padding_idx=0)
            self.char_rnn = nn.LSTM(input_size = opt['char_dim'],
                                    hidden_size = opt['char_hidden_size'],
                                    batch_first = True,
                                    bidirectional = True,
                                    num_layers = 1,
                                    dropout = 0)
            doc_input_size += 2 * opt['char_hidden_size']
            question_input_size += 2 * opt['char_hidden_size']


        self.cove_rnn = MTLSTM(opt,
                               embedding=embedding)
        self.word_attention_layer = WordAttention(input_size = opt['embedding_dim'],
                                                  hidden_size = opt['attention_size'],
                                                  dropout = opt['dropout_emb'],
                                                  device= self.device)

        self.low_doc_rnn = StackedLSTM(input_size=doc_input_size,
                                       hidden_size=opt['hidden_size'],
                                       num_layers=1,
                                       dropout=opt['dropout'],
                                       device= self.device)

        self.low_ques_rnn = StackedLSTM(input_size=question_input_size,
                                        hidden_size=opt['hidden_size'],
                                        num_layers=1,
                                        dropout=opt['dropout'],
                                        device= self.device)

        # Output sizes of low rnn encoders
        high_doc_hidden_size = 2 * opt['hidden_size']
        high_ques_hidden_size = 2 * opt['hidden_size']

        self.high_doc_rnn = StackedLSTM(input_size=high_doc_hidden_size,
                                        hidden_size=opt['hidden_size'],
                                        num_layers=1,
                                        dropout=opt['dropout'],
                                        device= self.device)

        self.high_ques_rnn = StackedLSTM(input_size=high_ques_hidden_size,
                                         hidden_size=opt['hidden_size'],
                                         num_layers=1,
                                         dropout=opt['dropout'],
                                         device= self.device)

        und_q_word_size = 2 * (2 * opt['hidden_size'])

        self.und_ques_rnn = StackedLSTM(input_size=und_q_word_size,
                                        hidden_size=opt['hidden_size'],
                                        num_layers=1,
                                        dropout=opt['dropout'],
                                        device= self.device)

        attention_inp_size = opt['embedding_dim'] + opt['cove_dim']+ 2 * (2 * opt['hidden_size'])

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
                                    num_layers = 1,
                                    dropout = opt['dropout'],
                                    device=self.device)

        self_attention_inp_size = opt['embedding_dim'] + \
                                  opt['cove_dim'] + \
                                  opt['pos_dim'] + opt['ner_dim']+ \
                                  6 * (2 * opt['hidden_size']) + 1


        self.self_attention_layer = FullAttention(input_size=self_attention_inp_size,
                                                  hidden_size=opt['attention_size'],
                                                  dropout=opt['dropout'],
                                                  device=self.device)

        self.self_rnn = StackedLSTM(input_size = 2 * (2 * opt['hidden_size']),
                                    hidden_size = opt['hidden_size'],
                                    num_layers = 1,
                                    dropout = opt['dropout'],
                                    device=self.device)

        self.summ_layer = Summ(input_size=2 * opt['hidden_size'],
                               dropout=opt['dropout'],
                               device=self.device)

        self.pointer_layer = PointerNet(input_size=2 * opt['hidden_size'],
                                        dropout=opt['dropout'],
                                        device=self.device)


    def forward(self, x1, x1_char, x1_pos, x1_ner, x1_origin, x1_lower, x1_lemma, x1_tf, x1_mask,
                x2, x2_char, x2_pos, x2_ner, x2_mask, x1_elmo=None, x2_elmo=None):

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

        ### GloVe ###
        x1_emb = self.embedding(x1)
        x2_emb = self.embedding(x2)
        ### CoVe ###
        _, x1_cove = self.cove_rnn(x1, x1_mask)
        _, x2_cove = self.cove_rnn(x2, x2_mask)

        ### embeddings ###
        x1_pos_emb = self.pos_embeddings(x1_pos)
        x1_ner_emb = self.ner_embeddings(x1_ner)
        x2_pos_emb = self.pos_embeddings(x2_pos)
        x2_ner_emb = self.ner_embeddings(x2_ner)

        ### embedding dropout ###
        x1_emb = Dropout(x1_emb, self.opt['dropout_emb'], self.training, device = self.device)
        x2_emb = Dropout(x2_emb, self.opt['dropout_emb'], self.training, device = self.device)
        x1_cove = Dropout(x1_cove, self.opt['dropout_emb'], self.training, device = self.device)
        x2_cove = Dropout(x2_cove, self.opt['dropout_emb'], self.training, device = self.device)
        x1_pos_emb = nn.functional.dropout(x1_pos_emb, p=self.opt['dropout_emb'], training=self.training)
        x1_ner_emb = nn.functional.dropout(x1_ner_emb, p=self.opt['dropout_emb'], training=self.training)

        word_attention_outputs = self.word_attention_layer(x1_emb, x1_mask, x2_emb, x2_mask, self.training)
        x1_word_input = torch.cat(
            [x1_emb, x1_cove, x1_pos_emb, x1_ner_emb, x1_tf, word_attention_outputs, x1_origin, x1_lower, x1_lemma], dim=2)
        x2_word_input = torch.cat([x2_emb, x2_cove, x2_pos_emb, x2_ner_emb], dim=2)

        if self.opt['use_char'] :
            x1_word_input = torch.cat([x1_word_input, x1_char_states], dim=2)
            x2_word_input = torch.cat([x2_word_input, x2_char_states], dim=2)

        ### low, high, understanding encoding ###
        low_x1_states = self.low_doc_rnn(x1_word_input, self.training)
        low_x2_states = self.low_ques_rnn(x2_word_input, self.training)

        high_x1_states = self.high_doc_rnn(low_x1_states, self.training)
        high_x2_states = self.high_ques_rnn(low_x2_states, self.training)

        und_x2_input = torch.cat([low_x2_states, high_x2_states], dim=2)
        und_x2_states = self.und_ques_rnn(und_x2_input, self.training)

        ########################################################################################


        ### Full Attention ###

        x1_How = torch.cat([x1_emb, x1_cove, low_x1_states, high_x1_states], dim=2)
        x2_How = torch.cat([x2_emb, x2_cove, low_x2_states, high_x2_states], dim=2)

        low_attention_outputs = self.low_attention_layer(x1_How, x1_mask, x2_How, x2_mask, low_x2_states, self.training)
        high_attention_outputs = self.high_attention_layer(x1_How, x1_mask, x2_How, x2_mask, high_x2_states, self.training)
        und_attention_outputs = self.und_attention_layer(x1_How, x1_mask, x2_How, x2_mask, und_x2_states, self.training)

        fuse_inp = torch.cat([low_x1_states, high_x1_states, low_attention_outputs, high_attention_outputs, und_attention_outputs], dim = 2)

        fused_x1_states = self.fuse_rnn(fuse_inp, self.training)

        ### Self Full Attention ###

        x1_How = torch.cat([x1_emb, x1_cove, x1_pos_emb, x1_ner_emb, x1_tf,
                            low_x1_states, high_x1_states,
                            low_attention_outputs, high_attention_outputs,
                            und_attention_outputs, fused_x1_states], dim=2)

        self_attention_outputs = self.self_attention_layer(x1_How, x1_mask, x1_How, x1_mask, fused_x1_states, self.training)

        self_inp = torch.cat([fused_x1_states, self_attention_outputs], dim=2)

        und_doc_states = self.self_rnn(self_inp, self.training)

        ### ques summ vector ###
        init_states = self.summ_layer(und_x2_states, x2_mask, self.training)

        ### Pointer Network ###
        logits1, logits2 = self.pointer_layer.forward(und_doc_states, x1_mask, init_states, self.training)
        if self.training:
            # In training we output log-softmax for NLL
            logits1 = F.log_softmax(logits1,dim=1)
            logits2 = F.log_softmax(logits2,dim=1)
        else:
            # ...Otherwise 0-1 probabilities
            logits1 = F.softmax(logits1,dim=1)
            logits2 = F.softmax(logits2, dim=1)

        return logits1, logits2


