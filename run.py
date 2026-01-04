import json
import os
import logging
import matplotlib.pyplot as plt
from transformers import AutoConfig, AutoModel, TrainingArguments
from models.PV_ORC import MyModel, MyConfig
from train import Mytrainer
from data_provider.data_loader import Dataset_Custom
from argparse import ArgumentParser
import random
import torch
import numpy as np
from utils.metrics import metric
from transformers.utils import logging as hf_logging

logger = hf_logging.get_logger(__name__)

file_handle = logging.FileHandler(filename='log.txt', mode='w')
file_handle.setLevel(logging.INFO)
file_handle.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

if len(logger.handlers) == 0: # 防止重复添加
    logger.addHandler(file_handle)
hf_logging.set_verbosity_info()

AutoConfig.register("MYMODEL", MyConfig)
AutoModel.register(MyConfig, MyModel)

parser = ArgumentParser()
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--pred_len', type=int, default=96, help='start token length')
parser.add_argument('--label_len', type=int, default=0, help='prediction sequence length')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcnh')
parser.add_argument('--llm_dim', type=int, default='768', help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--llm_layers', type=int, default=6)
parser.add_argument('--prompt_domain', type=int, default=0, help='')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--enc_in', type=int, default=17, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=17, help='decoder input size')
parser.add_argument('--c_out', type=int, default=17, help='output size')
parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--oenc_dim', type=int, default=128, help='feature flag')
parser.add_argument('--freq', type=str, default='h', help='feature flag')
parser.add_argument('--dataset_name', type=str, default='mydata_nx.csv', help='dataset name') 
parser.add_argument('--istrain', type=bool, default=True, help='train_of') 
args = parser.parse_args()

config = MyConfig(args=args)
model = MyModel(config)

fix_seed = 2026 
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

def compute_metrics(eval_preds):
    print("DEBUG: compute_metrics 正在运行！")
    predictions, labels = eval_preds
    labels = labels[:,:,-1][:,:,np.newaxis]
    # loss = ((predictions - labels) ** 2).mean()
    mae, mse, rmse, mape, mspe ,r2= metric(predictions, labels)
    logger.info(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}, MSPE: {mspe:.4f}, R2:{r2:.4f}")
    return {"eval_mae": mae, "eval_mse": mse, "eval_rmse": rmse, "eval_mape": mape, "eval_mspe": mspe, "r2": r2}

def no_op_preprocess(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0] # 取出 tuple 中的 tensor
    return logits

training_args = TrainingArguments(
    output_dir="result/",
    learning_rate=3.407619053379751e-05,
    lr_scheduler_type="cosine",
    per_device_train_batch_size=256,
    per_device_eval_batch_size=256,
    num_train_epochs=10,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="best",
    load_best_model_at_end=True,
    push_to_hub=False,
    # fp16=True, #newaddd---------------------------------------
    save_safetensors=False,
    # logging_dir="logs/",
    # logging_steps=50,
    remove_unused_columns=False,
    gradient_accumulation_steps=1,
    metric_for_best_model="eval_mae",
    greater_is_better=False,
    dataloader_drop_last=True,
    label_names=["labels"],
)


strain_dataset = Dataset_Custom(args=None,root_path="/EXP/PV_Prchetrator/LLMFM/data/", size=[args.seq_len,args.label_len,args.pred_len], features='MS', data_path=args.dataset_name,freq='h') 
test_dataset = Dataset_Custom(args=None,root_path="/EXP/PV_Prchetrator/LLMFM/data/", flag='test', size=[args.seq_len,args.label_len,args.pred_len], features='MS', data_path=args.dataset_name,freq='h') 
val_dataset = Dataset_Custom(args=None,root_path="/EXP/PV_Prchetrator/LLMFM/data/", flag='val', size=[args.seq_len,args.label_len,args.pred_len], features='MS', data_path=args.dataset_name,freq='h') 
trainer = Mytrainer(
    model=model,
    args=training_args,
    train_dataset = strain_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    # processing_class=tokenizer,
    preprocess_logits_for_metrics=no_op_preprocess,
    # data_collator=data_collator,
)

out ,metric_mse = trainer.train()
final_dir = "/home/qihui/EXP/PV_Prchetrator/final_model/NX_96"
trainer.model.save_pretrained(final_dir,safe_serialization=False)

trainer.model.llm.tokenizer.save_pretrained(final_dir)

args_dict = trainer.args.to_dict()

with open(os.path.join(final_dir, "training_args.json"), 'w', encoding='utf-8') as f:
    json.dump(args_dict, f, indent=4, ensure_ascii=False)