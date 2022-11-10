from more_itertools import chunked
import numpy as np
import pandas as pd
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
        
        self.modes = [
            (['qid', 'query', 'docno', self.text_field], self._transform_R), # encode both query and documents
            (['qid', 'query'], self._transform_Q), # query encoder
            (['docno', self.text_field], self._transform_D), # document encoder
#             (['docno','query', 'docno', 'query_vec',self.text_field], self._transform_R), 
        ]

    def transform(self, inp):
        #TODO: if quer_vec alreadyt exists in the column, how to retrieve documents using this query_vec?
        columns = set(inp.columns)
        
        for fields, fn in self.modes:
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
        if 'query_vec' in query_df.columns:
            print("query vector encoded")
#             reps = np.stack(query_df['query_vec'])
            query_reps = query_df['query_vec']
            texts = list(query_df[self.text_field])
            doc_reps = self._encode_texts(texts)
    
        else:
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
    
    

def _load_model(args, checkpoint_path):
    from ance.drivers.run_ann_data_gen import load_model
    # support downloads of checkpoints
    if checkpoint_path.startswith("http"):
        print("Downloading checkpoint %s" % checkpoint_path)
        import tempfile, wget
        targetZip = os.path.join(tempfile.mkdtemp(), 'checkpoint.zip')
        wget.download(checkpoint_path, targetZip)
        checkpoint_path = targetZip
    
    # support zip files of checkpoints
    if checkpoint_path.endswith(".zip"):
        import tempfile, zipfile
        print("Extracting checkpoint %s" % checkpoint_path)
        targetDir = tempfile.mkdtemp()
        zipfile.ZipFile(checkpoint_path).extractall(targetDir)
        #todo fix this
        checkpoint_path = os.path.join(targetDir, "Passage ANCE(FirstP) Checkpoint")

    print("Loading checkpoint %s" % checkpoint_path)
    config, tokenizer, model = load_model(args, checkpoint_path)
    return config, tokenizer, model
    
    
class ANCEPRF(ANCE):
    """
    in the following is the query encoder: ANCEPRF
    """
    def __init__(self, prf_type = "anceprf",k=3, model_name='/nfs/xiao/ANCE_PRF/pyterrier_ance/k3_checkpoint',cuda=None):
        super().__init__(model_name=model_name)
        self.k = k
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
        assert self.text_field in inp.columns

        new_qids = []
        new_query_embs = []
        iter = inp.groupby("qid")
        iter = pt.tqdm(iter, desc='PRF', unit='q') if self.verbose else iter
        for qid, group in iter:
            k = min(self.k, len(group))
            passage_texts = group.sort_values("rank").head(k)[self.text_field].values
            passage_texts = [group.iloc[0].query] + passage_texts
            #this line from pyserini
            full_text = f'{self.tokenizer.cls_token}{self.tokenizer.sep_token.join(passage_texts)}{self.tokenizer.sep_token}'
            
            new_qmeb = self.encode(full_text)
            new_qids.append(qid)
            new_query_embs.append( np.squeeze(new_qmeb) )
        qembs_df = pd.DataFrame(data={'qid' : new_qids, 'query_vec' : new_query_embs})
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
    
