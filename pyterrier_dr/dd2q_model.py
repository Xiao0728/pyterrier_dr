from more_itertools import chunked
import numpy as np
import pandas as pd
import torch
from torch import nn
import ir_datasets
import pyterrier as pt
from transformers import RobertaConfig, AutoTokenizer, AutoModel, AdamW
from torch.utils.tensorboard import SummaryWriter
from pyterrier_dr import ANCE, ANCEPRF
import argparse
logger = ir_datasets.log.easy()

parser = argparse.ArgumentParser()
parser.add_argument('--save-path', default = './checkpoint', type=str, help = 'save checkpoint path' )
parser.add_argument('--evalRes-path', default = './res-tct', type=str, help = 'save eval res path' )
parser.add_argument('--in_batch_negs', default = False, action='store_true')
parser.add_argument('--bsize', default=8, type=int, help='train batch size')

class DD2Q(pt.Transformer):
    def __init__(self, model_name='castorini/tct_colbert-msmarco', num_ref=80, batch_size=32, text_field='text', verbose=False, cuda=None):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.cuda = torch.cuda.is_available() if cuda is None else cuda
        self.model = AutoModel.from_pretrained(model_name).eval()
        if self.cuda:
            self.model = self.model.cuda()
        self.batch_size = batch_size
        self.text_field = text_field
        self.verbose = verbose
        self._optimizer = None
        self.modes = [
            (['qid', 'query', 'docno', self.text_field], self._transform_R),
            (['qid', 'query'], self._transform_Q),
            (['docno', self.text_field], self._transform_D),
        ]
        self.num_ref = num_ref

    def transform(self, inp):
        columns = set(inp.columns)
        
        for fields, fn in self.modes:
            if all(f in columns for f in fields):
                return fn(inp)
        message = f'Unexpected input with columns: {inp.columns}. Supports:'
        for fields, fn in self.modes:
            f += f'\n - {fn.__doc__.strip()}: {columns}\n'
        raise RuntimeError(message)
   
  
    
    def _encode_queries(self, texts):
        """
        Query vectorisation: use ANCE query encoder: input: reference shape:[batch_size x num_ref] output: ref_embs, shape: [batch_size x num_ref * dim (768)]  
        """
        results = []
        with torch.no_grad():
            for chunk in chunked(texts, self.batch_size):
                inps = self.tokenizer([f'[CLS] [Q] {q} ' + ' '.join(['[MASK]'] * 32) for q in chunk], add_special_tokens=False, return_tensors='pt', padding=True, truncation=True, max_length=36)
                if self.cuda:
                    inps = {k: v.cuda() for k, v in inps.items()}
                results = self.model(**inps).cpu().numpy()

        return np.concatenate(results, axis=0)


    def _encode_docs(self, texts):
        """
        Doc vectorisation: use TCT doc encoder?
        """
        results = []
        with torch.no_grad():
            for chunk in chunked(texts, self.batch_size):
                doc_text =['[CLS] '* self.num_ref +' [SEP] '  + f' {d}' for d in chunk]
                print(">>>>> doc_text",doc_text)
                inps = self.tokenizer(doc_text, add_special_tokens=False, return_tensors='pt', padding=True, truncation=True, max_length=512)
                
                if self.cuda:
                    inps = {k: v.cuda() for k, v in inps.items()}

                res = self.model(**inps).last_hidden_state
                res = res[:, :self.num_ref,:] # only takes the first num_prf cls embeddings as the doc_view embeddings to learn
                
                
                # TODO: do we need the below codes?
                res = res * inps['attention_mask'][:, :self.num_ref].unsqueeze(2) # apply attention mask
                lens = inps['attention_mask'][:, :self.num_ref].sum(dim=1).unsqueeze(1) 
                lens[lens==0]=1 
                res = res.sum(dim=1) / lens # average based on dim
                results.append(res.cpu().numpy())

        return np.concatenate(results, axis=0)
    def forward_score(self, refs, docs, return_matrix=False):
        r_embs = self._encode_queries(refs) #[BATCH_SIZE, NUM_REF, DIM]
        d_embs = self._encode_docs(docs) #[BATCH_SIZE, NUM_REF [MASK]S, DIM]
        matrix = torch.einsum('rie, die', r_embs, d_embs)
        scores_row_wise= matrix.max(dim=2).values.sum(dim=1) # for a given column,i.e. the given reference, obtain the maximum scored documents
        scores_col_wise = matrix.max(dim=2).values.sum(dim=0) # for a given row, i.e. the given document [MASK] view, obtain the maxium scored reference embedding;
        if return_matrix:
            return scores_row_wise, scores_col_wise, matrix
        return scores_row_wise, scores_col_wise




    def _transform_D(self, inp):
        """
        Document vectorisation
        """
        res = self._encode_docs(inp[self.text_field])
        return inp.assign(doc_vec=[res[i] for i in range(res.shape[0])])

    def _transform_Q(self, inp):
        """
        Query vectorisation
        """
        it = inp['query']
        if self.verbose:
            it = pt.tqdm(it, desc='Encoding Queies', unit='query')
        res = self._encode_queries(it)
        return inp.assign(query_vec=[res[i] for i in range(res.shape[0])])

    def _transform_R(self, inp):
        """
        Result re-ranking
        """
        return pt.apply.by_query(self._transform_R_byquery, add_ranks=True, verbose=self.verbose)(inp)

    def _transform_R_byquery(self, query_df):
        query_rep = self._encode_queries([query_df['query'].iloc[0]])
        doc_reps = self._encode_docs(query_df[self.text_field])
        scores = (query_rep * doc_reps).sum(axis=1)
        query_df['score'] = scores
        return query_df

    

        
