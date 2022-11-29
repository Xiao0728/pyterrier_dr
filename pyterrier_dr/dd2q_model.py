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
    def __init__(self, model_name='castorini/tct_colbert-msmarco', batch_size=32, text_field='text', verbose=False, cuda=None):
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
        

    def transform(self, inp):
        columns = set(inp.columns)
        
        for fields, fn in self.modes:
            if all(f in columns for f in fields):
                return fn(inp)
        message = f'Unexpected input with columns: {inp.columns}. Supports:'
        for fields, fn in self.modes:
            f += f'\n - {fn.__doc__.strip()}: {columns}\n'
        raise RuntimeError(message)
    # TODO: change the training code fit function:
    """
    1. there are there encoders, (a) query encoder, (b). doc-encoder, (c) reference d2qquery encoder
    2. structure: encoder (c) produce the d2q embeddings: in some way as what our baseline did: e.g. anceprf encoder?
    3. then, (a) query encoder encode the queries, (b) doc encoder encode the documents; 
    4. design the function using: MinSim (embeddings from (c), embeddings from (b));
    5. train the model, report the validation performance on dl2019.
    question: 
    i: which query encoder are we going to use as (a) and (b)? ANCE or TCT-ColBERT?
    ii: do we need hard-queries for training?

    """
    
    def _encode_queries(self, texts):
        """
        Query vectorisation: use TCT query encoder?
        """
        results = []
        with torch.no_grad():
            for chunk in chunked(texts, self.batch_size):
                inps = self.tokenizer([f'[CLS] [Q] {q} ' + ' '.join(['[MASK]'] * 32) for q in chunk], add_special_tokens=False, return_tensors='pt', padding=True, truncation=True, max_length=36)
                if self.cuda:
                    inps = {k: v.cuda() for k, v in inps.items()}
                res = self.model(**inps).last_hidden_state
                res = res[:, 4:, :].mean(dim=1) # remove the first 4 tokens (representing [CLS] [ Q ]), and average
                results.append(res.cpu().numpy())
        return np.concatenate(results, axis=0)

    def _encode_docs(self, texts):
        """
        Doc vectorisation: use TCT doc encoder?
        """
        results = []
        with torch.no_grad():
            for chunk in chunked(texts, self.batch_size):
                inps = self.tokenizer([f'[CLS] [D] {d}' for d in chunk], add_special_tokens=False, return_tensors='pt', padding=True, truncation=True, max_length=512)
                if self.cuda:
                    inps = {k: v.cuda() for k, v in inps.items()}
                res = self.model(**inps).last_hidden_state
                res = res[:, 4:, :] # remove the first 4 tokens (representing [CLS] [ D ])
                res = res * inps['attention_mask'][:, 4:].unsqueeze(2) # apply attention mask
                lens = inps['attention_mask'][:, 4:].sum(dim=1).unsqueeze(1)
                lens[lens == 0] = 1 # avoid edge case of div0 errors
                res = res.sum(dim=1) / lens # average based on dim
                results.append(res.cpu().numpy())
        return np.concatenate(results, axis=0)

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

    

        
        
class TCTPRF(TctColBert):
    """
    in the following is the query encoder: TCTPRF
    """
    def __init__(self, prf_type = "anceprf",k=3, return_docs=False,model_name='/nfs/xiao/TCT_PRF/checkpoint',cuda=None):
        super().__init__(model_name=model_name)
        self.k = k
        self.return_docs = return_docs

        if prf_type == "anceprf":
            """
            anceprf: a new encoder: input = [query + 3PRF doc_texts], as new query
            """
            self.modes = [
                (['qid', 'query', 'docno', self.text_field], self._transform_R_PRF)
            ]
        elif prf_type == "vectorprf":
            """
            vectorprf: take the average 
            """
            pass
        elif prf_type == "rocchioprf":
            """
            take the self.alpha*query_vec + 
            """
            pass

    def _transform_R_PRF(self, inp):
        assert "docno" in inp.columns
        assert "qid" in inp.columns
        assert "query" in inp.columns 
        assert "text" in inp.columns
        assert self.text_field in inp.columns

        new_qids = []
        new_query_embs = []
        iter = inp.groupby("qid")
        iter = pt.tqdm(iter, desc='PRF', unit='q') if self.verbose else iter
        for qid, group in iter:
            k = min(self.k, len(group))
            passage_texts = group.sort_values("rank").head(k)[self.text_field].values
            passage_texts = [group.iloc[0].query] + passage_texts
            #this line from APR: https://github.com/ielab/APR/blob/2b193113db7a50e666f31614b4ce1f1580f72212/pyserini/pyserini/dsearch/_prf.py#L310
            full_text =  f'{self.tokenizer.sep_token.join(passage_texts)}'
            #this line from pyserini
#             full_text = f'{self.tokenizer.cls_token}{self.tokenizer.sep_token.join(passage_texts)}{self.tokenizer.sep_token}'
            
            new_qmeb = self._encode_queries(full_text)
            new_qids.append(qid)
            new_query_embs.append( np.squeeze(new_qmeb) )
        qembs_df = pd.DataFrame(data={'qid' : new_qids, 'query_vec' : new_query_embs})

        if self.return_docs:
            rtr = inp[["qid","query","docno","text"]].merge(qembs_df,on='qid')
        else:
            rtr = inp[["qid", "query"]].drop_duplicates().merge(qembs_df, on='qid')        
        return rtr    

    

    def encode(self, prf_query : str):    
        import transformers
        inputs = self.tokenizer(
            [prf_query],
            max_length=512,
            padding='longest',
            truncation=True,
            add_special_tokens=False,
            return_tensors='pt'
        )  
       #inputs = inputs.to(cuda)
        inputs = inputs.to(self.model.device)
        embeddings = self.model(inputs["input_ids"], inputs["attention_mask"]).detach().cpu().numpy()
        return embeddings
    
