from inferflow.engine.llm_engine import LLMEngine


class LLM(LLMEngine):
    def __new__(cls, model: str, **kwargs):
        data_parallel_size = kwargs.get("data_parallel_size", 1)
        if data_parallel_size > 1:
            from inferflow.engine.dp_engine import DataParallelLLM

            return DataParallelLLM(model, **kwargs)
        return super().__new__(cls)
