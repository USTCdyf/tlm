from tqdm import tqdm
import numpy as np
import torch
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedConfig,
    AutoConfig,
    PreTrainedModel,
)
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

    def create_tokenizer(self) -> "TimeLLMTokenizer":

        return MeanScaleQuantileBins(**self.tokenizer_kwargs, config=self)


class TimeLLMTokenizer:
    """Base class for time series tokenizers"""

    def context_input_transform(self, context: torch.Tensor) -> Tuple:
        raise NotImplementedError()

    def label_input_transform(self, label: torch.Tensor, tokenizer_state: Any) -> Tuple:
        raise NotImplementedError()

    def output_transform(
        self, samples: torch.Tensor, tokenizer_state: Any
    ) -> torch.Tensor:
        raise NotImplementedError()


class MeanScaleQuantileBins(TimeLLMTokenizer):
    """Quantile-based binning tokenizer for time series"""

    def __init__(self, low_limit: float, high_limit: float, config: TimeLLMConfig):
        self.config = config
        self.centers = torch.linspace(
            low_limit,
            high_limit,
            config.n_tokens - 1,
        )
        self.boundaries = torch.concat(
            [
                torch.tensor([-1e20]),
                (self.centers[1:] + self.centers[:-1]) / 2,
                torch.tensor([1e20]),
            ]
        )

    def _input_transform(
        self, context: torch.Tensor, scale: Optional[torch.Tensor] = None
    ):
        context = context.float()
        attention_mask = ~torch.isnan(context)

        if scale is None:
            scale = torch.nansum(
                torch.abs(context) * attention_mask, dim=-1
            ) / torch.nansum(attention_mask, dim=-1)
            scale[~(scale > 0)] = 1.0

        scaled_context = context / scale.unsqueeze(-1)
        token_ids = torch.bucketize(scaled_context, self.boundaries, right=True).clamp(
            0, self.config.n_tokens - 1
        )  # 直接使用完整token空间

        return token_ids, attention_mask, scale

    def context_input_transform(self, context: torch.Tensor):
        # if context.shape[-1] > self.config.context_length:
        #     context = context[..., -self.config.context_length:]

        token_ids, attention_mask, scale = self._input_transform(context)

        return token_ids, attention_mask, scale

    def output_transform(self, samples: torch.Tensor, scale: torch.Tensor):
        """将模型输出的token索引转换为实际数值"""
        indices = torch.clamp(
            samples, min=0, max=len(self.centers) - 1  # 直接使用原始token索引
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
            key=x,  # [B, T, D]
            value=x,  # [B, T, D]
            key_padding_mask=key_padding_mask,  # 忽略padding部分
        )
        return attn_out  # [B, L, D]


class TimeLLMModel(PreTrainedModel):
    config_class = TimeLLMConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # 2. 文本处理模块
        self.llm_config = AutoConfig.from_pretrained(
            config.llm_name, trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.llm_name, trust_remote_code=True
        )
        self.llm = AutoModelForCausalLM.from_config(
            config=self.llm_config,
            # device_map='auto', # 必须用关键字参数
            trust_remote_code=True,
        ).half()
        self.llm_dim = self.llm_config.hidden_size  # 模型的隐藏维度

        # 1. 时序处理模块
        self.bin_embed = nn.Embedding(config.n_tokens, self.llm_dim)
        self.query_attn = QueryAttention(self.llm_dim, config.query_len)  # Query向量
        self.alignment = nn.Sequential(
            nn.Linear(self.llm_dim, self.llm_dim),
            nn.GELU(),
            nn.Linear(self.llm_dim, self.llm_dim),
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
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels
        )
        return outputs

    # def generate(self, bin_ids, text_input, max_new_tokens=500, eos_token_id=None):
    #     device = next(self.parameters()).device

    #     ts_emb = self.process_ts(bin_ids.to(device))  # [batch, 36, dim]

    #     # 2. 处理文本输入（需对齐时序长度变化）
    #     inputs = self.tokenizer(text_input, return_tensors='pt')
    #     input_ids = inputs.input_ids.to(device)

    #     # 关键修改：调整mask初始长度
    #     ts_len = ts_emb.shape[1]  # 36
    #     # text_len = input_ids.shape[1]
    #     attention_mask = torch.cat([
    #         torch.ones(1, ts_len, device=device),  # 时序部分全1
    #         inputs.attention_mask.to(device)       # 文本部分原始mask
    #     ], dim=1)

    #     # 3. 增量生成
    #     past_key_values = None
    #     generated_ids = input_ids.clone()

    #     print(f"时序长度: {ts_emb.shape[1]}, 文本长度: {input_ids.shape[1]}")
    #     print(self.llm.config.max_position_embeddings)

    #     for _ in range(max_new_tokens):
    #         # 文本嵌入（仅最新token）
    #         text_emb = self.llm.get_input_embeddings()(input_ids[:, -1:])
    #         inputs_embeds = torch.cat([ts_emb, text_emb], dim=1)

    #         # 动态扩展mask（每次新增1个token）
    #         current_mask = torch.cat([
    #             attention_mask,
    #             torch.ones(1, 1, device=device, dtype=torch.long)
    #         ], dim=1)

    #         outputs = self.llm(
    #             inputs_embeds=inputs_embeds,
    #             attention_mask=current_mask,
    #             past_key_values=past_key_values,
    #             use_cache=True
    #         )

    #         # 更新状态
    #         past_key_values = outputs.past_key_values
    #         next_token = outputs.logits[:, -1, :].argmax(-1)
    #         generated_ids = torch.cat([generated_ids, next_token.unsqueeze(-1)], dim=-1)
    #         attention_mask = current_mask  # 重要！保持mask同步更新

    #         # print("EOS token ID:", llm_tokenizer.eos_token_id)
    #         # print("首个生成token:", next_token.item())

    #         if eos_token_id is not None and next_token.item() == eos_token_id:
    #             break

    #     return self.tokenizer.decode(generated_ids[:, input_ids.shape[1]:][0], skip_special_tokens=True)
    def generate(self, bin_ids, text_input, max_new_tokens=500, eos_token_id=None):
        device = next(self.parameters()).device

        # 1. 时序编码（固定36长度）
        ts_emb = self.process_ts(bin_ids.to(device))  # [1,36,dim]

        # 2. 文本输入处理
        inputs = self.tokenizer(text_input, return_tensors="pt")
        input_ids = inputs.input_ids.to(device)
        text_emb = self.llm.get_input_embeddings()(input_ids)  # [1,text_len,dim]

        # 3. 拼接完整输入（时序+文本）
        inputs_embeds = torch.cat([ts_emb, text_emb], dim=1)  # [1,36+text_len,dim]
        attention_mask = torch.cat(
            [
                torch.ones(1, ts_emb.shape[1], device=device),
                inputs.attention_mask.to(device),
            ],
            dim=1,
        )

        # 4. 禁用缓存的全量生成
        generated_ids = input_ids.clone()
        for _ in range(max_new_tokens):
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,  # 关键修改
            )

            next_token = outputs.logits[:, -1, :].argmax(-1)
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(-1)], dim=-1)

            # 更新输入（包含所有历史token）
            new_emb = self.llm.get_input_embeddings()(next_token.unsqueeze(-1))
            inputs_embeds = torch.cat([inputs_embeds, new_emb], dim=1)  # 逐步增长
            attention_mask = torch.cat(
                [attention_mask, torch.ones(1, 1, device=device)], dim=1
            )

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

        return self.tokenizer.decode(
            generated_ids[:, input_ids.shape[1] :][0], skip_special_tokens=True
        )


# 1. 准备输入数据
x = {
    "instruction": "The historical load data is: 7633.8,7361.0,7136.0,6840.9,6624.9,6481.7,6426.5,6431.3,6466.9,6616.1,6686.2,6963.8,7234.6,7545.8,7819.5,8102.2,8342.4,8582.6,8781.7,8959.6,9173.4,9426.9,9619.0,9802.6,10005.1,10136.4,10342.1,10521.1,10716.8,10942.5,10901.9,10754.9,10724.1,10498.2,10434.6,10252.5,10187.4,9940.8,9693.8,9558.5,9444.0,9253.6,8946.8,8820.5,8444.4,8325.2,8043.2,7845.5",
    "input": "Based on the historical load data, please predict the load consumption in the next day. The region for prediction is NSW. The start date of historical data was on 2019-1-2 that is Weekday, and it is not a public holiday. The data frequency is 30 minutes per point. Historical data covers 1 day. The date of prediction is on 2019-1-3 that is Weekday, and it is not a public holiday. Weather of the start date: the minimum temperature is 294.54; the maximum temperature is 302.04; the humidity is 64.0; the pressure is 1013.0.  Weather of the prediction date: the minimum temperature is 293.03; the maximum temperature is 306.13; the humidity is 69.0; the pressure is 1013.0. There is no suitable news for long-term effect on future load consumption.In 2019-01-02 18:59:00, the news that had the Short-Term Effect on Today's Load Consumption is that Tropical cyclone Penny has reformed in the Coral Sea with weather forecasters predicting the storm will head back towards the east coast. The region that may be influenced is QLD. The rationality is that The presence of a tropical cyclone could lead to short-term fluctuations in load consumption as areas prepare for potential power outages or increased usage from emergency services and storm preparations. However, actual consumption might be reduced if outages occur and interrupt supply.In 2019-01-02 20:27:00, the news that had the Real-Time Direct Effect on Today's Load Consumption is that Darwin has sweated through its driest December since 1991, the Bureau of Meteorology has revealed. The region that may be influenced is WA. The rationality is that Given the dry conditions, there may be a higher usage of air conditioning and cooling systems leading to an immediate increase in electricity load consumption. The news report corresponds with the timeframe for possible real-time effects.",
    "output": "7667.5,7422.3,7222.8,6955.2,6726.2,6619.1,6569.4,6529.2,6522.6,6625.2,6721.6,6987.9,7206.5,7525.7,7877.8,8157.3,8338.1,8538.9,8596.9,8756.0,8877.0,8986.0,9161.8,9276.8,9477.4,9576.9,9614.9,9747.6,9893.6,10106.7,10268.2,10440.9,10540.1,10449.6,10354.6,10127.3,9999.4,9855.6,9656.4,9553.0,9499.1,9297.6,9002.2,8845.9,8633.8,8438.8,8193.0,7961.8",
}

# 2. 初始化tokenizer和配置
llm_tokenizer = AutoTokenizer.from_pretrained(
    "/home/scb123/HuggingfaceWeight/Qwen2.5-1.5B-Instruct"
)
config = TimeLLMConfig(
    tokenizer_class="MeanScaleQuantileBins",
    tokenizer_kwargs={"low_limit": -15.0, "high_limit": 15.0},
    prediction_length=48,
    n_tokens=4096,
    query_len=36,
    llm_name="/home/scb123/HuggingfaceWeight/Qwen2.5-1.5B-Instruct",
)

# 3. 数据预处理
# 时序数据编码
ts_values = [float(v) for v in x["instruction"].split(":")[1].strip().split(",")]
print(ts_values)
bin_tokenizer = config.create_tokenizer()
bin_ids, _, scale = bin_tokenizer.context_input_transform(torch.tensor([ts_values]))
print(bin_ids)

# 文本token化
# input_text = f"{x['input']}\nanswer:"
messages = [
    {
        "role": "system",
        "content": "Based on the historical one-day electricity load time series, as well as the future weather and time information, predict the electricity load changes in the specified region for the next day. Please output exactly 96 data points only, and nothing else.",
    },
    {"role": "user", "content": x["input"]},
]

full_text = llm_tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=False
)

# 4. 加载模型（假设模型已训练保存）
model = TimeLLMModel.from_pretrained(
    "/home/scb123/PyProject/tlm/dyf-feature_ds_config/final_model"
).to("cuda")
# print(model.llm.lm_head.weight.mean())
# first_layer = model.llm.model.layers[0]
# print(first_layer.mlp.up_proj.weight.mean())
model.eval()


with torch.no_grad():
    generated = model.generate(
        bin_ids=bin_ids,
        text_input=full_text,
        max_new_tokens=500,
        eos_token_id=llm_tokenizer.eos_token_id,
    )
print("预测结果:", generated)
