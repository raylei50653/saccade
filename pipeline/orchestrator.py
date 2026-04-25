import asyncio
import time
import uuid
from typing import Dict, Any, List, Tuple, Optional, cast, Awaitable
from storage.chroma_store import ChromaStore
from storage.redis_cache import RedisCache
from pipeline.health import HealthChecker, render
from dotenv import load_dotenv

# LlamaIndex 整合
try:
    from llama_index.core import VectorStoreIndex, StorageContext, Settings
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    # 預設使用 Ollama 作為本地 LLM
    from llama_index.llms.ollama import Ollama
    HAS_LLAMA_INDEX = True
except ImportError:
    HAS_LLAMA_INDEX = False

load_dotenv()


class PipelineOrchestrator:
    """
    Saccade 高效能編排器 (Micro-batching Edition)

    接收 YOLO 的高頻事件，將結構化數據轉化為語義記憶，存入 ChromaDB。
    整合 LlamaIndex 提供 Agentic RAG 能力。
    """

    def __init__(self) -> None:
        self.redis_cache = RedisCache()
        self.memory_store = ChromaStore()

        # 由於拔除 VLM，並發控制只需限制資料庫寫入頻率
        self.semaphore = asyncio.Semaphore(32)
        
        self.rag_engine: Optional[Any] = None
        if HAS_LLAMA_INDEX:
            self._setup_rag()

    def _setup_rag(self) -> None:
        """初始化 LlamaIndex RAG 引擎"""
        try:
            # 1. 設定本地模型
            Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
            Settings.llm = Ollama(model="llama3", request_timeout=30.0)

            # 2. 連結 ChromaDB
            vector_store = ChromaVectorStore(chroma_collection=self.memory_store.collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # 3. 建立索引與查詢引擎
            self.index = VectorStoreIndex.from_vector_store(
                vector_store, storage_context=storage_context
            )
            self.query_engine = self.index.as_query_engine(streaming=False)
            
            # 4. 註冊 Visual Re-query 工具
            from llama_index.core.tools import FunctionTool
            from llama_index.core.agent import ReActAgent
            
            def visual_requery(track_id: int) -> str:
                """
                When the LLM needs to visually confirm an object, it calls this tool with the track_id.
                It fetches the visual embedding (SigLIP) and searches ChromaDB for past appearances.
                """
                print(f"👁️ [Visual Re-query] Initiated for Track ID {track_id}")
                # 實務上應從 Redis (saccade:feat:{track_id}) 或共用記憶體拉取 FeatureBank 的特徵
                # 這裡為雛型展示，假設我們從 ChromaDB 的 hybrid_query 進行純向量搜尋
                # 假設我們已經拿到 embedding (這裡用 dummy vector 模擬)
                dummy_embedding = [0.0] * 768
                
                results = self.memory_store.hybrid_query(query_embedding=dummy_embedding, n_results=3)
                
                if not results or not results.get("documents") or not results["documents"][0]:
                    return f"No visual matches found for track {track_id}."
                
                matches = results["documents"][0]
                return f"Visual Re-query found {len(matches)} past appearances: {matches}"

            self.visual_tool = FunctionTool.from_defaults(fn=visual_requery)
            
            # 建立包含視覺重查能力的 Agent
            self.agent = ReActAgent.from_tools(
                [self.visual_tool], 
                llm=Settings.llm, 
                verbose=True
            )
            
            print("🚀 [Orchestrator] LlamaIndex RAG Agent Initialized with Visual Re-query.")
        except Exception as e:
            print(f"⚠️ [RAG Setup Error] {e}")

    def _generate_scene_description(self, objects: List[str], entropy: float) -> str:
        """基於 YOLO 標籤生成結構化的場景描述"""
        if not objects:
            return "Empty scene."

        obj_counts: Dict[str, int] = {}
        for obj in objects:
            obj_counts[obj] = obj_counts.get(obj, 0) + 1

        desc_parts = []
        for obj, count in obj_counts.items():
            desc_parts.append(f"{count} {obj}{'s' if count > 1 else ''}")

        base_desc = "Scene contains: " + ", ".join(desc_parts) + "."
        if entropy > 0.8:
            base_desc += " High dynamic activity detected."
        return base_desc

    async def _trigger_rag_analysis(self, query: str) -> None:
        """執行非同步 RAG 分析並輸出結果"""
        if not HAS_LLAMA_INDEX or not hasattr(self, "agent"):
            return

        print(f"🔍 [RAG Agent Query] {query}")
        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, self.agent.chat, query)
            print(f"🤖 [Agent Insight] {response}")
        except Exception as e:
            print(f"❌ [RAG Agent Error] {e}")

    async def handle_cognitive_event(self, event_data: Dict[str, Any]) -> None:
        """處理 YOLO 觸發的事件並持久化"""
        async with self.semaphore:
            metadata = event_data.get("metadata", {})
            frame_id = metadata.get("frame_id", 0)
            entropy = metadata.get("entropy_value", 0.0)
            yolo_objects = metadata.get("objects", [])

            # 1. 生成場景描述
            scene_description = self._generate_scene_description(yolo_objects, entropy)

            # 2. 定義異常邏輯 (基於規則)
            risk_objects = {"knife", "gun", "fire", "smoke", "accident"}
            is_anomaly = any(obj.lower() in risk_objects for obj in yolo_objects)

            # 3. 寫入 ChromaDB
            try:
                self.memory_store.add_memory(
                    content=scene_description,
                    metadata={
                        "frame_id": frame_id,
                        "entropy": entropy,
                        "objects": ", ".join(yolo_objects),
                        "is_anomaly": 1 if is_anomaly else 0,
                        "timestamp": time.time(),
                    },
                )
                status_tag = "🚨" if is_anomaly else "✅"
                # print(f"{status_tag} [Frame {frame_id}] Indexed: {scene_description}")
                
                # 4. 觸發 RAG 查詢 (當發生異常或複雜場景時)
                if HAS_LLAMA_INDEX and (is_anomaly or entropy > 0.9):
                    query = f"The current scene has high entropy ({entropy:.2f}) and contains {yolo_objects}. Are there any similar patterns in the past 5 minutes?"
                    asyncio.create_task(self._trigger_rag_analysis(query))

            except Exception as e:
                print(f"❌ [Storage Error] {e}")

    async def start_cognition_loop(self) -> None:
        print(
            "🚀 [Orchestrator] High-Speed Stream Indexing Loop Active (Micro-batching)..."
        )
        await self.redis_cache.connect()

        while True:
            try:
                # 1. 從 Redis Stream 撈取批次
                batch = await self.redis_cache.read_stream_batch(count=self.batch_size)

                if batch:
                    # 2. 異步處理該批次
                    asyncio.create_task(self.process_event_batch(batch))
                else:
                    # 沒資料時稍微休息
                    await asyncio.sleep(0.1)
            except Exception as e:
                print(f"⚠️ [Loop Error] {e}")
                await asyncio.sleep(1)

    async def run(self) -> None:
        checker = HealthChecker()
        report = await checker.run()
        print(render(report))

        try:
            await self.start_cognition_loop()
        finally:
            await self.redis_cache.disconnect()


async def main() -> None:
    orchestrator = PipelineOrchestrator()
    await orchestrator.run()


if __name__ == "__main__":
    asyncio.run(main())
