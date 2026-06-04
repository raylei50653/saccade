import torch
from transformers import AutoProcessor, AutoModel
from typing import Any, List, cast


class SigLIP2TextEncoder:
    """
    SigLIP2 文本編碼器 (Open-vocabulary 搜尋核心)
    """

    def __init__(
        self, model_id: str = "google/siglip2-base-patch16-224", device: str = "cuda"
    ):
        self.device = device
        print(f"⏳ Loading SigLIP2 Text Model from {model_id}...")
        self.model = AutoModel.from_pretrained(model_id).to(self.device).text_model
        self.processor = cast(Any, AutoProcessor).from_pretrained(model_id)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: List[str]) -> torch.Tensor:
        """
        將文本轉換為 Embedding 向量
        """
        inputs = self.processor(
            text=texts, padding="max_length", return_tensors="pt"
        ).to(self.device)
        text_features: torch.Tensor = self.model(**inputs).pooler_output
        # 正規化以利 Cosine Similarity 計算
        return cast(
            torch.Tensor, text_features / text_features.norm(dim=-1, keepdim=True)
        )


if __name__ == "__main__":
    encoder = SigLIP2TextEncoder()
    queries = ["a person checking phone", "a person walking", "a red car"]
    embeddings = encoder.encode(queries)
    print(f"✅ Text Embeddings Shape: {embeddings.shape}")
