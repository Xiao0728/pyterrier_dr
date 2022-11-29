import pyterrier as pt
if not pt.started():
    pt.init()
from pyterrier.measures import *
# from pyterrier_sbert import TctColBert, ColBERT
from pyterrier_dr import ANCE, ANCEPRF
from pyterrier_dr import DD2Q
from transformers import AdamW
import torch
from torch import nn
import ir_datasets
from more_itertools import chunked
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import argparse

logger = ir_datasets.log.easy()
parser = argparse.ArgumentParser()
parser.add_argument('--save-path', default = './checkpoint', type=str, help = 'save checkpoint path' )
parser.add_argument('--evalRes-path', default = './res-tct', type=str, help = 'save eval res path' )
parser.add_argument('--in_batch_negs', default = False, action='store_true')
parser.add_argument('--bsize', default=8, type=int, help='train batch size')

temperature = 0.25
# in_batch_negs = True
# batch_size =8
def processQuery(query):
    query = re.sub(r"[^a-zA-Z0-9¿]+", " ", query)
    return query

def clean(inputDF):
    inputDF['query']=inputDF['query'].apply(lambda query: processQuery(query))
    return inputDF




def main():
    args = parser.parse_args()
    print("args.bsize",args.bsize)
    print("in_batch_negs:",args.in_batch_negs)
    if args.in_batch_negs:
        writer = SummaryWriter(f"/nfs/sean/workspace_xiao/pyterrier_sbert/tensorboard/bsize{args.bsize}-tct_ibn_devt100/")
    else: 
        writer = SummaryWriter(f"/nfs/sean/workspace_xiao/pyterrier_sbert/tensorboard/bsize{args.bsize}-tct_pair_devt100/") 

    # colbert_model = ColBERT('./colbert-uog', modeltype='orig')
    # tct_model = TctColBert('bert-base-uncased', verbose=True) # lets firstly assume that we use TCTColBER
    ref_model = ANCEPRF('./anceprf checkpoint') # use anceprf model as the reference model
    dd2q_model  = DD2Q('./dd2q', verbose=True)
    

    optimizer = AdamW([p for p in dd2q_model.model.parameters() if p.requires_grad], lr=1e-5, eps=1e-8)
    optimizer.zero_grad()
    # loss_fn = nn.KLDivLoss(reduction='batchmean')
    loss_fn_ce = nn.CrossEntropyLoss()
    loss_fn_mse = nn.MSELoss()


    # dataset = pt.get_dataset('irds:msmarco-passage/train/triples-small')
    # 
    dataset = pt.get_dataset('irds:msmarco-passage/train/split200-train')
    irds = dataset.irds_ref()
    # valid_dataset = pt.get_dataset('irds:msmarco-passage/train/split200-valid')
    # For trec 2019 to evaluate
    # valid_dataset = pt.get_dataset('irds:msmarco-passage/trec-dl-2019/judged')
    # valid_irds = valid_dataset.irds_ref()
    # # valid_data = pd.DataFrame(valid_irds.scoreddocs).rename(columns={'query_id': 'qid', 'doc_id': 'docno'})
    # # valid_data = (pt.apply.query_text(lambda x: valid_irds.queries.lookup(x['qid']).text) >> pt.text.get_text(valid_dataset, 'text'))(valid_data)
    # # valid_data = valid_data.rename(columns={'query_text': 'query'})


    # For DEV evaluation
    
    # dev_dataset = pt.get_dataset('irds:msmarco-passage/dev/small')
    # irds = dataset.irds_ref()
    # dev_irds = dataset.irds_ref()
    # dev_data = pd.DataFrame(dev_irds.scoreddocs).rename(columns={'query_id': 'qid', 'doc_id': 'docno'})
    # dev_qrels = dev_dataset.get_qrels()
    # dev_data_t100 = (dev_data.head(200)).merge(dev_qrels[dev_qrels["label"]>0][["qid"]].drop_duplicates()).head(100)
    # dev_data_t100 = (pt.apply.query_text(lambda x: dev_irds.queries.lookup(x['qid']).text) >> pt.text.get_text(dev_dataset, 'text'))(dev_data_t100)
    # dev_data_t100 = dev_data_t100.rename(columns={'query_text': 'query'})

    ref_model.model.eval()
    pair_it = iter(irds.docpairs)
    tct_ndcg_max =0.0
    for outer_i in logger.pbar(range(500), desc='validation loop'):
        running_losses, running_accs = [], []
        dd2q_model.model.train()
        with logger.pbar_raw(desc='train loop', total=1000) as pbar:
            for i, chunk in enumerate(chunked(pair_it, args.bsize//2)):
                if i == 1000:
                    break
                batch_q, batch_d, batch_ref = [], [], [] 
                for qid, pos_did, neg_did in chunk:
                    q = irds.queries.lookup(qid).text
                    batch_q.append(q)
                    if not args.in_batch_negs:
                        batch_q.append(q)
                    batch_d.append(irds.docs.lookup(pos_did).text)
                    batch_d.append(irds.docs.lookup(neg_did).text)
                    # for each training query, we retrieve from the d2q index, and obtain the text from dataset; function to merge them 
                    # obtain the full_text: ref_query + PRF texts:
                    # encode by the PRF encoder;
                    # 

                    ref_qs = d2q_dict['pos_did'].split(".")
                    for q in ref_qs:
                        batch_ref.append(q)
                    batch_ref.append(irds.docs.lookup(pos_did).text) # batch_r #TODO-1: how to obtain d2q generated queries, by given the pos/neg document ids?

                # with torch.no_grad():

                #     colbert_scores = colbert_model.model.forward_score(batch_q, batch_d, all_pairs=args.in_batch_negs)
                # if args.in_batch_negs:
                #     colbert_scores = colbert_scores
                # else:
                #     colbert_scores = colbert_scores.reshape(-1, 2)
                
                # colbert_probs = (colbert_scores/temperature).softmax(dim=1)
            
                # this line of code is wrong
                batch_ref = ref_model.tokenizer([f'[CLS] [Q] {q} ' + ' '.join(['[MASK]'] * 32) for q in batch_ref], add_special_tokens=False, return_tensors='pt', padding=True, truncation=True, max_length=512)
                batch_q = dd2q_model.tokenizer([f'[CLS] [Q] {q} ' + ' '.join(['[MASK]'] * 32) for q in batch_q], add_special_tokens=False, return_tensors='pt', padding=True, truncation=True, max_length=36)
                batch_d = dd2q_model.tokenizer([f'[CLS] [D] {d}' for d in batch_d], add_special_tokens=False, return_tensors='pt', padding=True, truncation=True, max_length=256)
                batch_q = {k: v.cuda() for k, v in batch_q.items()}
                batch_d = {k: v.cuda() for k, v in batch_d.items()}
                batch_ref = {k: v.cuda() for k,v in batch_ref.items()}


                R = ref_model.model(**batch_ref).last_hiddent_state
                R = R[:, 4:, :].mean(dim=1) # same with the TCT-PRF/or ANCE-PRF 



                Q = dd2q_model.model(**batch_q).last_hidden_state
                Q = Q[:, 4:, :].mean(dim=1) # remove the first 4 tokens (representing [CLS] [ Q ]), and average

                D = dd2q_model.model(**batch_d).last_hidden_state
                
                loss_minsim = loss_fn_mse(R,D)
                




                D_ce = D[:, 4:, :] # remove the first 4 tokens (representing [CLS] [ D ])
                D_ce = D_ce * batch_d['attention_mask'][:, 4:].unsqueeze(2) # apply attention mask
                D_ce = D_ce.sum(dim=1) / batch_d['attention_mask'][:, 4:].sum(dim=1).unsqueeze(1) # average based on dim


                if args.in_batch_negs:
                    scores = torch.einsum('qe,de->qd', Q, D_ce)
                    # targets = (torch.arange(batch_q['input_ids'].shape[0]) * 2).to(tct_scores.device)
                else:
                    scores = torch.einsum('be,be->b', Q, D_ce).reshape(-1, 2)
                

                logprobs = nn.functional.log_softmax(scores, dim=1)
                loss_ce = loss_fn_ce(logprobs)
                loss = loss_minsim + loss_ce


                loss = loss.cpu().detach().item()
                optimizer.step()
                optimizer.zero_grad()
                pbar.update()
                running_losses.append(loss)
                
                pbar.set_postfix({'loss': loss})

                writer.add_scalar("train/1.loss", loss, outer_i)
                
               

                # if len(running_losses) == 100:
                    # logger.info(f'it={i+1} loss={sum(running_losses)/len(running_losses)} acc={sum(running_accs)/len(running_accs)}')
                    # running_losses, running_accs = [], []
            writer.add_scalar("train/3.runningloss", sum(running_losses[-1000:])/len(running_losses[-1000:]), outer_i)   
            


        dd2q_model.model.eval()
        # results = pt.Experiment([tct_model], valid_data, valid_dataset.get_qrels(), [nDCG@10, RR(rel=2), AP(rel=2), Judged@10])
        # topicsDev100 = pd.read_csv("./topicsDev_t100.txt")
        topicsDev100 = pd.read_csv("./split200_topics.txt")
        # topicsDev100.qid = topicsDev100.qid.astype(str)
        # topicsDev100.docno = topicsDev100.docno.astype(str)
        qrels = pd.read_csv("./qrels_split_200.txt")
        results = pt.Experiment([tct_model], topicsDev100, qrels, [nDCG@10, RR(rel=2), AP(rel=2), Judged@10]) 
        # results = pt.Experiment([tct_model], topicsDev100, pt.get_dataset("trec-deep-learning-passages").get_qrels('dev.small'),[nDCG@10, RR(rel=2), AP(rel=2), Judged@10])
        print(outer_i, results)

        tct_ndcg = results.iloc[0]["nDCG@10"]
        writer.add_scalar("eval/1.tct.ndcg10", tct_ndcg, outer_i)
        writer.flush()
        if results.iloc[0]["nDCG@10"] > tct_ndcg_max: 
            tct_ndcg_max = results.iloc[0]["nDCG@10"]
            print(f" New Max NDCG@10 score reached at epoch{outer_i} with NDCG@10 score = {tct_ndcg_max}") 
            if args.in_batch_negs:
                dd2q_model.model.save_pretrained(f'/nfs/sean/workspace_xiao/pyterrier_sbert/checkpoint/tct-ibn-b{args.bsize}_dev/tct-ibn-{outer_i}-{tct_ndcg_max}')
                dd2q_model.tokenizer.save_pretrained(f'/nfs/sean/workspace_xiao/pyterrier_sbert//checkpoint/tct-ibn-b{args.bsize}_dev/tct-ibn-{outer_i}-{tct_ndcg_max}') 
            else:
                dd2q_model.model.save_pretrained(f'/nfs/sean/workspace_xiao/pyterrier_sbert/checkpoint/tct-pair-b{args.bsize}_dev/tct-pair-{outer_i}-{tct_ndcg_max}')
                dd2q_model.tokenizer.save_pretrained(f'/nfs/sean/workspace_xiao/pyterrier_sbert/checkpoint/tct-pair-b{args.bsize}_dev/tct-pair-{outer_i}-{tct_ndcg_max}')
        if args.in_batch_negs:
            with open(f'./eval_res_26Jan/tct-ibn_bsize{args.bsize}_dev100.txt', 'at') as fout:
                fout.write(f'{outer_i} {args.bsize} {results}\n')
            # tct_model.model.save_pretrained(f'./checkpoint/tct-ibn/tct-ibn-{outer_i}')
            # tct_model.tokenizer.save_pretrained(f'./checkpoint/tct-ibn/tct-ibn-{outer_i}')
        else:
            with open(f'./eval_res_26Jan/tct-pair_bsize{args.bsize}_dev100.txt', 'at') as fout:
                fout.write(f'{outer_i} {args.bsize} {results}\n')
            # tct_model.model.save_pretrained(f'./checkpoint/tct-pair/tct-pair-{outer_i}')
            # tct_model.tokenizer.save_pretrained(f'./checkpoint/tct-pair/tct-pair-{outer_i}')


if __name__ == '__main__':
    main()