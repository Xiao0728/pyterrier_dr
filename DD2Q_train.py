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
    # loss_fn_ce = nn.CrossEntropyLoss()
    # loss_fn_mse = nn.MSELoss()

    loss_fn = nn.CrossEntropyLoss()
    # dataset = pt.get_dataset('irds:msmarco-passage/train/triples-small')
    # 
    dataset = pt.get_dataset('irds:msmarco-passage')
    irds = dataset.irds_ref()
   

    ref_model.model.eval()
    pair_it = iter(irds.docs) # contain: d
    num_ref = 80
    running_losses, running_accs= [],[]
    dd2q_model.model.train()
    with logger.pbar_raw(desc='train loop', total=1000) as pbar:
            for i, chunk in enumerate(chunked(pair_it, args.bsize)):
                if i == 1000:
                    break
                batch_ref, batch_d = [], [] 
                for doc in chunk:
                    doc_id = doc.doc_id
                    batch_d.append(doc.text)
                    
                    ref = references_dictionary(doc_id)# TODO: obtain all the references according to the current doc_id also control the number of reference for each docid.
                    batch_ref.append(ref)
                with torch.no_grad():
                    batch_ref = ref_model.tokenizer([f'{q}'+ ' '.join(['[MASK]'] * 32) for q in batch_ref],add_special_tokens=False, return_tensors='pt', padding=True, truncation=True, max_length=512) 
                    batch_ref = {k: v.cuda() for k,v in batch_ref.items()}
                    R = ref_model.model(**batch_ref).last_hidden_state

                # the numeber of [MASK] tokens = the number of references for each docid
                batch_d = dd2q_model.tokenizer(['[CLS] '* num_ref + f'{d}' for d in batch_d], add_special_tokens=False, return_tensors='pt', padding=True, truncation=True, max_length=512)
                batch_d = {k: v.cuda() for k, v in batch_d.items()}

                D = dd2q_model.model(**batch_d).last_hidden_state
                D = D[:, :num_ref, :] # take the first num_ref [CLS] embeddings as the view embeddings


                matrix = torch.einsum('rie, die', R, D)
                scores_row_wise= matrix.max(dim=2).values.sum(dim=1) # for a given column,i.e. the given reference, obtain the maximum scored documents
                scores_col_wise = matrix.max(dim=2).values.sum(dim=0) 


                # TODO: then we need targets for socres_row_wise, and scores_col_wise
                # TODO: then, we have two types of loss. the recall loss and the prevision loss, calculated using the 
                #  row_wise and col_wise, be careful here
                # TODO: the losses below are wrong, haven't thought them carefully

                targets_row_wise = torch.zeros_like(scores_row_wise[:, 0]).long() 
                loss_row_wise = loss_fn(scores_row_wise, targets_row_wise)
                # acc_row_wise = ((scores_row_wise.max(dim=1).indices == targets_row_wise).sum) / scores_row_wise.shape[0].cpu().detach().item()
                
                targets_col_wise = torch.zeros_like(scores_col_wise[:, 0]).long() 
                loss_col_wise = loss_fn(scores_col_wise, targets_col_wise)
                # acc_col_wise = ((scores_row_wise.max(dim=1).indices == targets_row_wise).sum) / scores_row_wise.shape[0].cpu().detach().item()
                
                loss = loss_row_wise + loss_col_wise



                loss.backward()
                loss = loss.cpu().detach().item()
                optimizer.step()
                optimizer.zero_grad()
                pbar.update()
                running_losses.append(loss)
                running_accs.append(acc)
                pbar.set_postfix({'loss': loss, 'acc': acc})
                if len(running_losses) == 100:
                    logger.info(f'it={i+1} loss={sum(running_losses)/len(running_losses)} acc={sum(running_accs)/len(running_accs)}')
                    running_losses, running_accs = [], []

                # writer.add_scalar("train/1.loss", loss, outer_i)
                
               

                # if len(running_losses) == 100:
                    # logger.info(f'it={i+1} loss={sum(running_losses)/len(running_losses)} acc={sum(running_accs)/len(running_accs)}')
                    # running_losses, running_accs = [], []
            writer.add_scalar("train/3.runningloss", sum(running_losses[-1000:])/len(running_losses[-1000:]), outer_i)   
            


if __name__ == '__main__':
    main()