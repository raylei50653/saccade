import os
from llama_cpp import Llama
from dotenv import load_dotenv

load_dotenv()

class LLMEngine:
    def __init__(self, model_path: str, n_gpu_layers: int = -1):
        """
        初始化 LLM 引擎 (llama-cpp-python)
        
        :param model_path: GGUF 模型路徑
        :param n_gpu_layers: offload 到 GPU 的層數 (-1 表示全部)
        """
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.llm = None
        
        if os.path.exists(model_path):
            self.load_model()
        else:
            print(f"Warning: Model file not found at {model_path}")

    def load_model(self):
        """載入 GGUF 模型"""
        self.llm = Llama(
            model_path=self.model_path,
            n_gpu_layers=self.n_gpu_layers,
            n_ctx=2048, # 預設上下文長度
            verbose=False
        )
        print(f"LLM loaded from {self.model_path} with {self.n_gpu_layers} GPU layers.")

    def generate(self, prompt: str, max_tokens: int = 128):
        """執行推理"""
        if not self.llm:
            return "Error: LLM model not loaded."
        
        output = self.llm(
            f"Q: {prompt} A: ",
            max_tokens=max_tokens,
            stop=["Q:", "\n"],
            echo=True
        )
        return output['choices'][0]['text']

if __name__ == "__main__":
    # 測試程式碼
    # engine = LLMEngine("path/to/model.gguf")
    # print(engine.generate("Describe the activity in the video stream."))
    pass
