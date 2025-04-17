from models.TLM import TimeLLMConfig

def get_time_llm_config():
    return TimeLLMConfig(
        tokenizer_class="MeanScaleQuantileBins",
        tokenizer_kwargs={"low_limit": -15.0, "high_limit": 15.0},
        prediction_length=48,
        n_tokens=4096,
        query_len=36,
        llm_name="Qwen/Qwen2.5-1.5B-Instruct"
    )