import random
import torch
import torch.nn as nn


class EncoderRNN(nn.Module):
    '''Encoder RNN for Seq2Seq'''
    def __init__(self, hidden_dim, emb_dim, vocab_size,  weights_pretrained, dropout = 0.2, n_layers = 1):
        '''
        params:
            :hidden_dim: hidden dimension of GRU
            :emb_dim: dimension of embedding trained on our corpus
            :vocab_size: size of vocabulary of pretrained embeddings
            :weights_pretrained: weights to load into Embeding
            :dropout: dropout fo GRU
            :n_layers: number of GRU layers
        '''
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.emb1 = nn.Embedding.from_pretrained(weights_pretrained)  # no autograd
        self.emb2 = nn.Embedding(vocab_size, emb_dim)  # autograd

        self.gru = nn.GRU(
            input_size=emb_dim+300,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, seq, hidden = None):
        x = self.emb1(seq)
        y = self.emb2(seq)
        x = torch.cat((x, y), dim=2)
        if hidden is None:
            hidden = torch.zeros(
                self.n_layers, seq.shape[0], self.hidden_dim, requires_grad=True)
        out, hidden = self.gru(x, hidden)
        return out, hidden
        

class DecoderRNN(nn.Module):
    '''Decoder for Seq2Seq'''
    def __init__(self, hidden_dim, emb_dim, vocab_size, weights_pretrained, output_dim=4, dropout=0.2, max_length=200, n_layers = 1):
        '''
        params:
            :hidden_dim: hidden dimension of GRU
            :emb_dim: dimension of embedding trained on our corpus
            :vocab_size: size of vocabulary of pretrained embeddings
            :weights_pretrained: weights to load into Embeding
            :output_dim: vocabulary size of target sequence
            :dropout: dropout fo GRU
            :n_layers: number of GRU layers
        '''
        super(DecoderRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.max_length = max_length
        self.n_layers = n_layers
        
        self.emb1 = nn.Embedding.from_pretrained(weights_pretrained)  # no autograd
        self.emb2 = nn.Embedding(vocab_size, emb_dim)  # autograd

        
        self.gru = nn.GRU(
            input_size=emb_dim+300,
            hidden_size=self.hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        
        self.dropout = nn.Dropout(self.dropout_p)
        self.embedding = nn.Embedding(5, 400, padding_idx=4)
        self.out = nn.Linear(self.hidden_dim, self.output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, seq, encoder_outputs, hidden = None):
        seq = seq.unsqueeze(0)
        
        embedded = self.embedding(seq)
        
        if hidden is None:
            hidden = torch.zeros(
                self.n_layers, seq.shape[0], self.hidden_dim, requires_grad=True)
        
        
        output, hidden = self.gru(embedded, hidden)
        pred = self.out(output.squeeze(0))
        return pred, hidden#, attn_weights


class Seq2Seq(nn.Module):
    '''Seq2seq model for aspect extraction'''
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hidden_dim == decoder.hidden_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        _, hidden = self.encoder(src)
        
        seq = torch.LongTensor(batch_size).fill_(3)
        for t in range(0, trg_len):
            
            output, hidden = self.decoder(seq, hidden)
            outputs[:,t,:] = output
            
            teacher_force = random.random() < teacher_forcing_ratio
            
            top1 = output.argmax(1) 
            
            seq =  trg[:,t] if teacher_force else top1
        
        return outputs