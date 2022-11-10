from more_itertools import chunked
import numpy as np
import torch
from torch import nn
import pyterrier as pt
from transformers import RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification


class ANCE(pt.transformer.TransformerBase):
    def __init__(self, model_name='/home/sean/data/ance/msmarco-firstp-checkpoint/', batch_size=216, text_field='text', verbose=False, cuda=None):
        self.model_name = model_name
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name, do_lower_case=True)
        self.cuda = torch.cuda.is_available() if cuda is None else cuda
        self.model = _ANCEModel.from_pretrained(model_name).eval()
        if self.cuda:
            self.model = self.model.cuda()
            self.cuda = True
        self.batch_size = batch_size
        self.text_field = text_field
        self.verbose = verbose

    def transform(self, inp):
        columns = set(inp.columns)
        modes = [
            (['qid', 'query', 'docno', self.text_field], self._transform_R),
            (['qid', 'query'], self._transform_Q),
            (['docno', self.text_field], self._transform_D),
        ]
        for fields, fn in modes:
            if all(f in columns for f in fields):
                return fn(inp)
        message = f'Unexpected input with columns: {inp.columns}. Supports:'
        for fields, fn in modes:
            f += f'\n - {fn.__doc__.strip()}: {columns}\n'
        raise RuntimeError(message)

    def _encode_texts(self, texts):
        results = []
        with torch.no_grad():
            for chunk in chunked(texts, self.batch_size):
                inps = self.tokenizer(chunk, add_special_tokens=True, return_tensors='pt', padding=True, truncation=True)
                # inps = []
                # for doc in chunk:
                #     inps.append(self.tokenizer.encode_plus(doc, add_special_tokens=True, return_tensors='pt'))
                # inps = {
                #     'input_ids': torch.cat([i['input_ids'] for i in inps]),
                #     'attention_mask': torch.cat([i['attention_mask'] for i in inps]),
                # }
                if self.cuda:
                    inps = {k: v.cuda() for k, v in inps.items()}
                results.append(self.model(**inps).cpu().numpy())
        return np.concatenate(results, axis=0)

    def _transform_D(self, inp):
        """
        Document vectorisation
        """
        res = self._encode_texts(inp[self.text_field])
        return inp.assign(doc_vec=[res[i] for i in range(res.shape[0])])

    def _transform_Q(self, inp):
        """
        Query vectorisation
        """
        res = self._encode_texts(inp['query'])
        return inp.assign(query_vec=[res[i] for i in range(res.shape[0])])

    def _transform_R(self, inp):
        """
        Result re-ranking
        """
        return pt.apply.by_query(self._transform_R_byquery, add_ranks=True, verbose=self.verbose)(inp)

    def _transform_R_byquery(self, query_df):
        texts = [query_df['query'].iloc[0]] + list(query_df[self.text_field])
        reps = self._encode_texts(texts)
        query_rep = reps[0:1] # keep batch dim; allows broadcasting
        doc_reps = reps[1:]
        scores = (query_rep * doc_reps).sum(axis=1)
        query_df['score'] = scores
        return query_df


class _ANCEModel(RobertaForSequenceClassification):
    def __init__(self, config, model_argobj=None):
        RobertaForSequenceClassification.__init__(self, config)
        self.embeddingHead = nn.Linear(config.hidden_size, 768)
        self.norm = nn.LayerNorm(768)
        self.apply(self._init_weights)

    def forward(self, input_ids, attention_mask):
        outputs1 = self.roberta(input_ids=input_ids,
                                attention_mask=attention_mask)
        full_emb = outputs1[0][:, 0] # get [CLS]
        query1 = self.norm(self.embeddingHead(full_emb))
        return query1