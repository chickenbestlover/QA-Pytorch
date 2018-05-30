# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import torch
import torch.nn as nn
from drqa import layers2 as layers
from drqa import multiAttentionRNN as custom
from cove import MTLSTM
# Modification: add 'pos' and 'ner' features.
# Origin: https://github.com/facebookresearch/ParlAI/tree/master/parlai/agents/drqa

def normalize_emb_(data):
    print (data.size(), data[:10].norm(2,1))
    norms = data.norm(2,1) + 1e-8
    if norms.dim() == 1:
        norms = norms.unsqueeze(1)
    data.div_(norms.expand_as(data))
    print (data.size(), data[:10].norm(2,1))

class RnnDocReader(nn.Module):
    """Network for the Document Reader module of DrQA."""
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    def __init__(self, opt, padding_idx=0, embedding=None, normalize_emb=False,embedding_order=True):
        super(RnnDocReader, self).__init__()
        # Store config
        self.opt = opt

        # Word embeddings
        if opt['pretrained_words']:
            assert embedding is not None
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

        else:  # random initialized
            self.embedding = nn.Embedding(opt['vocab_size'],
                                          opt['embedding_dim'],
                                          padding_idx=padding_idx)

        # Projection for attention weighted question
        if opt['use_qemb']:
            self.qemb_match = layers.SeqAttnMatch(opt['embedding_dim'])

        # Input size to RNN: word emb + question emb + manual features
        doc_input_size = opt['embedding_dim'] + opt['num_features']
        if opt['use_qemb']:
            doc_input_size += opt['embedding_dim']
        if opt['pos']:
            doc_input_size += opt['pos_size']
        if opt['ner']:
            doc_input_size += opt['ner_size']

        # Input size to RNN: word emb + question emb + manual features

        question_input_size = opt['embedding_dim']

        print('doc_input_size:',doc_input_size)
        self.attention_rnns= custom.AttentionRNN(opt,
                                                 doc_input_size=doc_input_size,
                                                 question_input_size=question_input_size,
                                                 ratio=opt['reduction_ratio'])

        # Output sizes of rnn encoders
        doc_hidden_size = 2 * opt['hidden_size'] +opt['hidden_size']//opt['reduction_ratio']
        question_hidden_size =  2 * opt['hidden_size']+opt['hidden_size']//opt['reduction_ratio']

        # Question merging
        if opt['question_merge'] not in ['avg', 'self_attn']:
            raise NotImplementedError('question_merge = %s' % opt['question_merge'])
        if opt['question_merge'] == 'self_attn':
            self.self_attn = layers.LinearSeqAttn(question_hidden_size)


        # Bilinear attention for span start/end
        self.start_attn = layers.BilinearSeqAttn(
            doc_hidden_size,
            question_hidden_size,
        )
        self.end_attn = layers.BilinearSeqAttn(
            doc_hidden_size,
            question_hidden_size,
        )

    def forward(self, x1, x1_f, x1_pos, x1_ner, x1_mask, x2, x2_mask):
        """Inputs:
        x1 = document word indices             [batch * len_d]
        x1_f = document word features indices  [batch * len_d * nfeat]
        x1_pos = document POS tags             [batch * len_d]
        x1_ner = document entity tags          [batch * len_d]
        x1_mask = document padding mask        [batch * len_d]
        x2 = question word indices             [batch * len_q]
        x2_mask = question padding mask        [batch * len_q]
        """
        # Embed both document and question
        x1_emb = self.embedding(x1)
        x2_emb = self.embedding(x2)

        if self.opt['dropout_emb'] > 0:
            x1_emb = nn.functional.dropout(x1_emb, p=self.opt['dropout_emb'],
                                               training=self.training)
            x2_emb = nn.functional.dropout(x2_emb, p=self.opt['dropout_emb'],
                                           training=self.training)
        drnn_input_list = [x1_emb, x1_f]
        # Add attention-weighted question representation
        if self.opt['use_qemb']:
            x2_weighted_emb = self.qemb_match(x1_emb, x2_emb, x2_mask)
            drnn_input_list.append(x2_weighted_emb)
        if self.opt['pos']:
            if self.opt['dropout_emb'] > 0:
                x1_pos = nn.functional.dropout(x1_pos, p=self.opt['dropout_emb'],
                                               training=self.training)
            drnn_input_list.append(x1_pos)
        if self.opt['ner']:
            if self.opt['dropout_emb'] > 0:
                x1_ner = nn.functional.dropout(x1_ner, p=self.opt['dropout_emb'],
                                               training=self.training)
            drnn_input_list.append(x1_ner)
        drnn_input = torch.cat(drnn_input_list, 2)
        #print('drnn_input:',drnn_input.size())
        # Encode document with RNN
        doc_hiddens, question_hiddens = self.attention_rnns(drnn_input,x1_mask,x2_emb,x2_mask)
        q_merge_weights = self.self_attn(question_hiddens, x2_mask)
        question_hidden = layers.weighted_avg(question_hiddens, q_merge_weights)

        start_scores = self.start_attn(doc_hiddens, question_hidden, x1_mask)
        end_scores = self.end_attn(doc_hiddens, question_hidden, x1_mask)
        return start_scores, end_scores
