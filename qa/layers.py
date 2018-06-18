import torch
import torch.nn as nn
import torch.nn.functional as F

import cuda_functional as custom_nn

class SRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1,dropout=0,bidirectional=True,batch_first=True):
        super(SRU, self).__init__()
        self.sru = custom_nn.SRUCell(n_in= input_size,
                                 n_out=hidden_size,
                                 dropout=dropout,
                                 rnn_dropout=0,
                                 use_tanh=1,
                                 use_selu=0,
                                 bidirectional=bidirectional,
                                 layer_norm=False,
                                 rescale=False)
    def forward(self, x):
        out, c  = self.sru.forward(x.transpose(0,1))
        return out.transpose(0,1), c.transpose(0,1)

class StackedLSTM(nn.Module) :

    def __init__(self, input_size, hidden_size, num_layers, dropout,
                 dropout_rnn=0,concat = False, device = 'cuda', rnn_type=nn.LSTM, res=False, norm=False) :
        super(StackedLSTM, self).__init__()
        self.device = device
        self.dropout = dropout
        self.concat = concat
        self.rnn_type =rnn_type
        self.num_layers = num_layers
        self.res =res
        self.norm=norm
        self.rnns = nn.ModuleList()
        if norm:
            self.norm = GroupNorm(2*hidden_size,num_group=4)

        for layer in range(num_layers) :
            self.rnns.append(rnn_type(input_size = input_size if layer == 0 else 2 * hidden_size,
                                     hidden_size = hidden_size if layer == 0 else 2* hidden_size,
                                     num_layers = 1,
                                     dropout = dropout_rnn,
                                     batch_first=True,
                                     bidirectional=True if layer == 0 else False))


    def forward(self, x) :
        x = Dropout(x, self.dropout, self.training, device=self.device)

        outputs = [x]

        for layer in range(self.num_layers) :
            inp = outputs[layer]
            if self.res and layer==1:
                inp_res = inp.clone()

            out, _ = self.rnns[layer](inp)

            if self.res and layer==self.num_layers-1:
                out = out+inp_res
            if self.norm and layer==self.num_layers-1:
                out = self.norm(out)
            outputs.append(out)

        outputs = outputs[1:]
        if self.concat :
            return torch.cat(outputs, dim=2)
        else :
            return outputs[-1]

class FullAttention(nn.Module) :

    def __init__(self, input_size, hidden_size, dropout, device = 'cuda'):
        super(FullAttention, self).__init__()
        self.device = device
        self.dropout = dropout
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.U = nn.Linear(input_size, hidden_size, bias=False)
        self.D = nn.Parameter(torch.ones(1, hidden_size), requires_grad=True)

        self.init_weights()

    def init_weights(self) :
        nn.init.xavier_uniform_(self.U.weight.data)

    def forward(self, passage, p_mask, question, q_mask, rep):

        if self.training:
            keep_prob = 1.0 - self.dropout
            drop_mask = Dropout(passage, self.dropout, self.training, return_mask=True, device = self.device)
            d_passage = torch.div(passage, keep_prob) * drop_mask
            d_ques = torch.div(question, keep_prob) * drop_mask
        else :
            d_passage = passage
            d_ques = question

        Up = F.relu(self.U(d_passage))
        Uq = F.relu(self.U(d_ques))
        D = self.D.expand_as(Uq)

        Uq = D * Uq

        scores = Up.bmm(Uq.transpose(2, 1))

        mask = q_mask.unsqueeze(1).repeat(1, passage.size(1), 1)
        scores.data.masked_fill_(mask.data, -float('inf'))
        alpha = F.softmax(scores, 2)
        output = torch.bmm(alpha, rep)

        return output

class FullAttention_multiHead(nn.Module) :

    def __init__(self, input_size, hidden_size, dropout, n_head=5, device = 'cuda'):
        super(FullAttention_multiHead, self).__init__()
        self.device = device
        self.dropout = dropout
        self.input_size = input_size
        self.hidden_size = hidden_size // n_head
        self.D = nn.Parameter(torch.ones(1, self.hidden_size), requires_grad=True)
        self.W = nn.Parameter(torch.randn(n_head, input_size, self.hidden_size,dtype=torch.float))
        self.U = nn.Parameter(torch.randn(n_head, hidden_size, self.hidden_size,dtype=torch.float))
        self.n_head = n_head
        self.V = nn.Linear(self.n_head*self.hidden_size, hidden_size)
        self.init_weights()

    def init_weights(self) :
        nn.init.xavier_uniform_(self.W.data)
        nn.init.xavier_uniform_(self.U.data)
        nn.init.xavier_uniform_(self.V.weight.data)

    def forward(self, passage, p_mask, question, q_mask, rep):

        if self.training:
            keep_prob = 1.0 - self.dropout
            drop_mask = Dropout(passage, self.dropout, self.training, return_mask=True, device = self.device)
            d_passage = torch.div(passage, keep_prob) * drop_mask
            d_ques = torch.div(question, keep_prob) * drop_mask
        else :
            d_passage = passage
            d_ques = question
        # Project vectors
        x_s = d_passage.repeat(self.n_head,1,1).view(self.n_head,-1,passage.size(2)) # n_head * (batch x len1) * input_size
        y_s = d_ques.repeat(self.n_head, 1, 1).view(self.n_head, -1, question.size(2)) # n_head * (batch x len2) * input_size
        x_s = torch.bmm(x_s,self.W) # n_head * (batch x len1) * hidden_size
        y_s = torch.bmm(y_s, self.W)  # n_head * (batch x len2) * hidden_size
        x_s = x_s.view(-1,passage.size(1),self.hidden_size) # (n_head x batch) * len1 * hidden_size
        y_s = y_s.view(-1, question.size(1), self.hidden_size) # (n_head x batch) * len2 * hidden_size
        x_s_proj = F.relu(x_s)
        y_s_proj = F.relu(y_s)
        D = self.D.expand_as(y_s_proj)
        y_s_proj = D * y_s_proj
        # Compute scores
        scores = x_s_proj.bmm(y_s_proj.transpose(2, 1)) # (n_head x batch) * len1 * len2

        # Mask padding
        mask = q_mask.unsqueeze(1).repeat(self.n_head,passage.size(1),1)
        #print('y_mask:',y_mask.size())
        #print('scores:', scores.size())

        scores.data.masked_fill_(mask.data, -float('inf'))

        # Normalize with softmax
        alpha_flat = F.softmax(scores.view(-1, question.size(1)),dim=1)
        alpha = alpha_flat.view(-1, passage.size(1), question.size(1))
        N,rep_len,f_size = rep.size()
        rep = rep.repeat(self.n_head,1,1).view(self.n_head,-1,f_size) # n_head * (batch x len1) * input_size
        rep = torch.bmm(rep,self.U)
        rep = rep.view(-1, rep_len,self.hidden_size) # (n_head x batch) * len1 * hidden_size
        rep = F.relu(rep)


        # Take weighted average
        matched_seq = alpha.bmm(rep) # (n_head x batch) * len1 * hidden_size
        matched_seq = torch.cat(torch.split(matched_seq,passage.size(0),dim=0),dim=-1) # batch x len1 x (n_head * hidden_size)
        matched_seq = F.relu(self.V(matched_seq)) # batch x len1 x output_size

        return matched_seq




class WordAttention(nn.Module) :

    def __init__(self, input_size, hidden_size, dropout, device = 'cuda') :
        super(WordAttention, self).__init__()
        self.device = device
        self.dropout = dropout
        self.hidden_size = hidden_size

        self.W = nn.Linear(input_size, hidden_size)
        self.init_weights()

    def init_weights(self) :
        nn.init.xavier_uniform_(self.W.weight.data)
        self.W.bias.data.fill_(0.1)

    def forward(self, passage, p_mask, question, q_mask):

        if self.training:
            keep_prob = 1.0 - self.dropout
            drop_mask = Dropout(passage, self.dropout, self.training, return_mask = True, device = self.device)
            d_passage = torch.div(passage, keep_prob) * drop_mask
            d_ques = torch.div(question, keep_prob) * drop_mask
        else :
            d_passage = passage
            d_ques = question

        Wp = F.relu(self.W(d_passage))
        Wq = F.relu(self.W(d_ques))

        scores = torch.bmm(Wp, Wq.transpose(2, 1))

        mask = q_mask.unsqueeze(1).repeat(1, passage.size(1), 1)
        scores.data.masked_fill_(mask.data, -float('inf'))
        alpha = F.softmax(scores, dim=2)
        output = torch.bmm(alpha, question)

        return output

class WordAttention_multiHead(nn.Module) :

    def __init__(self, input_size, hidden_size, dropout, n_head=5,device = 'cuda') :
        super(WordAttention_multiHead, self).__init__()
        self.device = device
        self.dropout = dropout
        self.hidden_size = input_size // n_head
        self.W = nn.Parameter(torch.randn(n_head, input_size, self.hidden_size,dtype=torch.float))
        self.n_head = n_head
        self.init_weights()

    def init_weights(self) :
        nn.init.xavier_uniform_(self.W.data)

    def forward(self, passage, p_mask, question, q_mask):

        if self.training:
            keep_prob = 1.0 - self.dropout
            drop_mask = Dropout(passage, self.dropout, self.training, return_mask = True, device = self.device)
            d_passage = torch.div(passage, keep_prob) * drop_mask
            d_ques = torch.div(question, keep_prob) * drop_mask
        else :
            d_passage = passage
            d_ques = question

        # Project vectors
        x_s = d_passage.repeat(self.n_head,1,1).view(self.n_head,-1,passage.size(2)) # n_head * (batch x len1) * input_size
        y_s = d_ques.repeat(self.n_head, 1, 1).view(self.n_head, -1, question.size(2)) # n_head * (batch x len2) * input_size
        x_s = torch.bmm(x_s,self.W) # n_head * (batch x len1) * hidden_size
        y_s = torch.bmm(y_s, self.W)  # n_head * (batch x len2) * hidden_size
        x_s = x_s.view(-1,passage.size(1),self.hidden_size) # (n_head x batch) * len1 * hhidden_size
        y_s = y_s.view(-1, question.size(1), self.hidden_size) # (n_head x batch) * len2 * hidden_size
        x_s_proj = F.relu(x_s)
        y_s_proj = F.relu(y_s)

        # Compute scores
        scores = x_s_proj.bmm(y_s_proj.transpose(2, 1)) # (n_head x batch) * len1 * len2

        # Mask padding
        mask = q_mask.unsqueeze(1).repeat(self.n_head,passage.size(1),1)
        #print('y_mask:',y_mask.size())
        #print('scores:', scores.size())

        scores.data.masked_fill_(mask.data, -float('inf'))

        # Normalize with softmax
        alpha_flat = F.softmax(scores.view(-1, question.size(1)),dim=1)
        alpha = alpha_flat.view(-1, passage.size(1), question.size(1))

        # Take weighted average
        matched_seq = alpha.bmm(y_s) # (n_head x batch) * len1 * hidden_size
        matched_seq = torch.cat(torch.split(matched_seq,passage.size(0),dim=0),dim=-1) # batch x len1 x (n_head * hidden_size)

        return matched_seq


class Summ(nn.Module) :

    def __init__(self, input_size, dropout, device = 'cuda') :
        super(Summ, self).__init__()
        self.device = device
        self.dropout = dropout
        self.w = nn.Linear(input_size, 1)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.w.weight.data)
        self.w.bias.data.fill_(0.1)

    def forward(self, x, mask) :

        d_x = Dropout(x, self.dropout, self.training, device = self.device)
        beta = self.w(d_x).squeeze(2)
        beta.data.masked_fill_(mask.data, -float('inf'))
        beta = F.softmax(beta, 1)
        output = torch.bmm(beta.unsqueeze(1), x).squeeze(1)
        return output

class PointerNet(nn.Module) :

    def __init__(self, input_size, dropout, device = 'cuda') :
        super(PointerNet, self).__init__()
        self.device = device
        self.dropout = dropout

        self.W_s = nn.Linear(input_size, input_size)
        self.W_e = nn.Linear(input_size, input_size)
        self.rnn = nn.GRUCell(input_size, input_size)

        self.init_weights()

    def init_weights(self) :

        nn.init.xavier_uniform_(self.W_s.weight.data)
        self.W_s.bias.data.fill_(0.1)
        nn.init.xavier_uniform_(self.W_e.weight.data)
        self.W_e.bias.data.fill_(0.1)

    def forward(self, self_states, p_mask, init_states) :

        d_init_states = Dropout(init_states.unsqueeze(1), self.dropout, self.training, device = self.device).squeeze(1)
        P0 = self.W_s(d_init_states)
        logits1 = torch.bmm(P0.unsqueeze(1), self_states.transpose(2, 1)).squeeze(1)
        logits1.data.masked_fill_(p_mask.data, -float('inf'))
        P_s = F.softmax(logits1, 1)

        rnn_input = torch.bmm(P_s.unsqueeze(1), self_states).squeeze(1)
        d_rnn_input = Dropout(rnn_input.unsqueeze(1), self.dropout, self.training, device = self.device).squeeze(1)
        end_states = self.rnn(d_rnn_input, init_states)
        d_end_states = Dropout(end_states.unsqueeze(1), self.dropout, self.training, device = self.device).squeeze(1)

        P1 = self.W_e(d_end_states)
        logits2 = torch.bmm(P1.unsqueeze(1), self_states.transpose(2, 1)).squeeze(1)
        logits2.data.masked_fill_(p_mask.data, -float('inf'))
        return logits1, logits2

def Dropout(x, dropout, is_train, return_mask = False, var=True, device='cuda') :

    if not var :
        return F.dropout(x, dropout, is_train)

    if dropout > 0.0 and is_train :
        shape = x.size()
        keep_prob = 1.0 - dropout
        random_tensor = keep_prob
        tmp = torch.FloatTensor(shape[0], 1, shape[2]).to(device)
        nn.init.uniform_(tmp)
        random_tensor += tmp
        binary_tensor = torch.floor(random_tensor)
        x = torch.div(x, keep_prob) * binary_tensor

    if return_mask :
        return binary_tensor


    return x

class GroupNorm(nn.Module):
    def __init__(self, hidden_size, num_group=4,eps=1e-6):
        super(GroupNorm, self).__init__()
        self.eps = eps
        self.num_group=num_group
        self.a = nn.Parameter(torch.ones(hidden_size), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(hidden_size), requires_grad=True)

    def forward(self, x):
        if x.size(-1) == 1:
            return x
        N,seq_len,H = x.size()
        x = x.contiguous().view(N*seq_len,self.num_group,-1)
        mu = torch.mean(x, dim=-1)
        sigma = torch.std(x, dim=-1, unbiased=False)
        # HACK. PyTorch is changing behavior
        if mu.dim() == x.dim()-1:
            mu = mu.unsqueeze(mu.dim())
            sigma = sigma.unsqueeze(sigma.dim())
        output = (x - mu.expand_as(x)) / (sigma.expand_as(x) + self.eps)
        output =output.view(N,seq_len,H)
        output = output.mul(self.a.expand_as(output)) \
            + self.b.expand_as(output)
        return output