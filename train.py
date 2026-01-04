import torch
import torch.nn as nn
from transformers import Trainer
from datasets import general_dataset
import pandas as pd
class Mytrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_losses = []
        self.return_eval_mse = 999
    def compute_loss(self,model,inputs,return_outputs=False, **kwargs):
        print("inputs keys:", inputs.keys())

        # 1. 提取标签
        # 从输入字典中把 'labels' 拿出来，因为我们后面要自己算 loss，不能让模型在 forward() 内部算
        # labels = inputs.pop('labels')
        labels = inputs["labels"]
        labels = labels[:,:,-1].unsqueeze(-1)
        # 2. 前向传播 (Forward Pass)
        # 把剩下的 inputs 喂给模型，得到输出

        
        outputs = model(**inputs)
        # 3. 获取 Logits
        # Logits 是模型输出的原始分数（未经过 Softmax 的概率）
        out = outputs.get("logits")
        loss_fct = nn.MSELoss()
        # 5. 计算损失
        # 将 logits 和 labels 调整形状后传入损失函数计算 Loss
        loss = loss_fct(out,labels)  # 应该是 loss_fct(outputs, labels)loss_fct(outputs, labels)
        # 6. 返回结果
        # 必须返回计算出来的 loss，如果 trainer 要求返回 outputs，也一并返回

        self.train_losses.append(loss.item())
        return (loss, out) if return_outputs else loss

    def create_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            betas=(0.9, 0.95),   # LLaMA/GPT 推荐值
            eps=1e-8,
            weight_decay=self.args.weight_decay,
        )

    def evaluation_loop(self, *args, **kwargs):
        """
        重写评估循环以确保 compute_metrics 被调用
        """
        output = super().evaluation_loop(*args, **kwargs)
        
        # 打印调试信息
        print(f"\n评估完成，指标: {output.metrics}")
        return output
    
    def train(self, *args, **kwargs):
        """
        重写训练方法以记录损失
        """
        output = super().train(*args, **kwargs)
        return output,self.state.best_metric

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # 1. 准备输入
        inputs = self._prepare_inputs(inputs)
        
        # 2. 计算 (不自动处理输出，直接拿 compute_loss 的结果)
        with torch.no_grad():
            if self.args.eval_use_gather_object:
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                logits = outputs
            else:
                loss, logits = self.compute_loss(model, inputs, return_outputs=True)

        if prediction_loss_only:
            return (loss, None, None)
            
        # 3. 这里的 logits 是完整的 [128, 24, 1]，直接返回，Trainer 也就无法切片了
        return (loss, logits, inputs.get("labels"))
    
    