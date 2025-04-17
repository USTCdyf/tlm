import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig, AutoConfig, PreTrainedModel
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

class TimeLLMConfig(PretrainedConfig):
    """
    HF-Compatible Configuration for TimeLLM Model
    (Non-dataclass version with full PretrainedConfig integration)
    """
    model_type = "time_llm"
    def __init__(
        self,
        tokenizer_kwargs: Dict[str, Any] = None,
        prediction_length: int = 24,
        n_tokens: int = 4096,
        query_len: int = 36,
        **kwargs
    ):
        # 必须调用父类初始化（处理HF标准参数）
        super().__init__(**kwargs)

        # 核心自定义参数
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        self.prediction_length = prediction_length
        self.n_tokens = n_tokens
        self.query_len = query_len

    def create_tokenizer(self) -> 'TimeLLMTokenizer':
        
        return MeanScaleQuantileBins(**self.tokenizer_kwargs, config=self)

class TimeLLMTokenizer:
    """Base class for time series tokenizers"""
    def context_input_transform(self, context: torch.Tensor) -> Tuple:
        raise NotImplementedError()
    
    def label_input_transform(self, label: torch.Tensor, tokenizer_state: Any) -> Tuple:
        raise NotImplementedError()
    
    def output_transform(self, samples: torch.Tensor, tokenizer_state: Any) -> torch.Tensor:
        raise NotImplementedError()

class MeanScaleQuantileBins(TimeLLMTokenizer):
    """Quantile-based binning tokenizer for time series"""
    def __init__(self, low_limit: float, high_limit: float, config: TimeLLMConfig):
        self.config = config
        self.centers = torch.linspace(
            low_limit, high_limit,
            config.n_tokens - 1,
        )
        self.boundaries = torch.concat([
            torch.tensor([-1e20]),
            (self.centers[1:] + self.centers[:-1]) / 2,
            torch.tensor([1e20])
        ])

    def _input_transform(self, context: torch.Tensor, scale: Optional[torch.Tensor] = None):
        context = context.float()
        attention_mask = ~torch.isnan(context)

        if scale is None:
            scale = torch.nansum(torch.abs(context) * attention_mask, dim=-1) / \
                   torch.nansum(attention_mask, dim=-1)
            scale[~(scale > 0)] = 1.0

        scaled_context = context / scale.unsqueeze(-1)
        token_ids = torch.bucketize(scaled_context, self.boundaries,right=True).clamp(0, self.config.n_tokens - 1)  # 直接使用完整token空间
        
        return token_ids, attention_mask, scale

    def context_input_transform(self, context: torch.Tensor):
        # if context.shape[-1] > self.config.context_length:
        #     context = context[..., -self.config.context_length:]
            
        token_ids, attention_mask, scale = self._input_transform(context)
            
        return token_ids, attention_mask, scale

    def output_transform(self, samples: torch.Tensor, scale: torch.Tensor):
        """将模型输出的token索引转换为实际数值"""
        indices = torch.clamp(
            samples,  # 直接使用原始token索引
            min=0,
            max=len(self.centers)-1
        )
        return self.centers[indices] * scale.unsqueeze(-1).unsqueeze(-1)


class QueryAttention(nn.Module):
    
    def __init__(self, embed_dim: int, latent_len: int, num_heads: int = 8):
        super().__init__()
        self.latent_len = latent_len
        # 可学习的Query矩阵 (L×D)
        self.query = nn.Parameter(torch.randn(latent_len, embed_dim))
        # 多头注意力
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        """
        输入: 
          x: [B, T, D] 
          key_padding_mask: [B, T]（可选）
        输出: 
          [B, L, D]
        """
        # 扩展Query为[B, L, D]
        queries = self.query.unsqueeze(0).expand(x.size(0), -1, -1)
        
        # 注意力计算
        attn_out, _ = self.attn(
            query=queries,  # [B, L, D]
            key=x,          # [B, T, D]
            value=x,        # [B, T, D]
            key_padding_mask=key_padding_mask  # 忽略padding部分
        )
        return attn_out  # [B, L, D]

class TimeLLMModel(PreTrainedModel):
    config_class = TimeLLMConfig
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        # 2. 文本处理模块
        self.llm_config = AutoConfig.from_pretrained(config.llm_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(config.llm_name, trust_remote_code=True)
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.llm_name,
            config=self.llm_config,
            # device_map='auto', # 必须用关键字参数
            trust_remote_code=True
        )
        for param in self.llm.parameters():
            param.requires_grad = False
        self.llm_dim = self.llm_config.hidden_size  # 模型的隐藏维度

        # 1. 时序处理模块
        self.bin_embed = nn.Embedding(config.n_tokens, self.llm_dim)
        self.query_attn = QueryAttention(self.llm_dim, config.query_len)  # Query向量
        self.alignment = nn.Sequential(
            nn.Linear(self.llm_dim, self.llm_dim),
            nn.GELU(),
            nn.Linear(self.llm_dim, self.llm_dim)
        )
        
    def process_ts(self, bin_id):
        """时序数据编码：分箱 -> Embedding -> 注意力筛选 -> 投影"""
        # ts_data: [batch, context_length]
        
        bin_embedding = self.bin_embed(bin_id)  # [batch, ctx_len, dim]
        bin_feat = self.query_attn(bin_embedding)
        
        return self.alignment(bin_feat)

    def forward(self, input_ids, attention_mask, bin_ids, labels=None, scales=None):
        """
        关键设计：
        - 输入格式：[时序Token][文本Token][预测值Token]
        - 训练时：通过错位labels实现自回归
        """
        # 1. 处理时序数据
        ts_emb = self.process_ts(bin_ids) 
        
        # 2. 获取文本嵌入
        text_emb = self.llm.get_input_embeddings()(input_ids)  
        
        # 3. 拼接输入 [时序][文本]
        inputs_embeds = torch.cat([ts_emb, text_emb], dim=1)  
        
        # 5. 通过LLM生成（自回归）
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels  
        )
        return outputs

    def generate(self, ts_data, text_input, max_new_tokens=48):
        # 初始化输入
        input_ids = self.tokenizer(text_input, return_tensors='pt').input_ids
        ts_emb = self.process_ts(ts_data)
        
        # 自回归循环
        for _ in range(max_new_tokens):
            text_emb = self.llm.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat([ts_emb, text_emb], dim=1)
            
            outputs = self.llm(inputs_embeds=inputs_embeds)
            next_token = outputs.logits[:, -1, :].argmax(-1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
            
        return input_ids[:, -max_new_tokens:]  

    
class TimeLLMPipeline:
    def __init__(self, config, model_path=None):
        self.config = config
        self.tokenizer = config.create_tokenizer()
        self.model = TimeLLMModel(config)
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path))

    def preprocess(self, raw_ts: List[float], text: str) -> Dict[str, torch.Tensor]:
        """将原始数据转换为模型输入"""
        # 1. 时序分箱
        bin_ids, _, scale = self.tokenizer.context_input_transform(
            torch.tensor([raw_ts])
        )
        
        # 2. 文本token化
        text_enc = self.model.tokenizer(
            text, 
            return_tensors="pt",
            padding="max_length",
            # max_length=self.config.context_length
        )
        
        return {
            "bin_ids": bin_ids,
            "input_ids": text_enc.input_ids,
            "attention_mask": text_enc.attention_mask,
            "scale": scale  # 保留用于逆变换
        }

    def postprocess(self, pred_tokens: torch.Tensor, scale: float) -> List[float]:
        """将模型输出转换为原始数值"""
        return self.tokenizer.output_transform(pred_tokens, scale).tolist()

    def predict(self, raw_ts: List[float], text: str) -> List[float]:
        """端到端预测"""
        inputs = self.preprocess(raw_ts, text)
        pred_tokens = self.model.generate(
            inputs["bin_ids"], 
            text_input=text,
            max_new_tokens=self.config.prediction_length
        )
        return self.postprocess(pred_tokens, inputs["scale"])
