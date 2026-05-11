import os
from typing import Dict, Any, List, Optional
from sqlalchemy import Column, Integer, Float, JSON, DateTime, ForeignKey, Text, select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func

Base = declarative_base()

class Experiment(Base):
    __tablename__ = "experiments"
    id = Column(Integer, primary_key=True)
    name = Column(Text, nullable=False)
    category = Column(Text, nullable=False)
    config = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    description = Column(Text)
    metrics = relationship("Metric", back_populates="experiment", cascade="all, delete-orphan")

class Metric(Base):
    __tablename__ = "metrics"
    id = Column(Integer, primary_key=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id", ondelete="CASCADE"))
    dataset = Column(Text, nullable=False)
    idf1 = Column(Float)
    recall = Column(Float)
    precision = Column(Float)
    mota = Column(Float)
    num_switches = Column(Integer)
    num_false_positives = Column(Integer)
    num_misses = Column(Integer)
    raw_results = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    experiment = relationship("Experiment", back_populates="metrics")

class AblationStore:
    """
    Saccade Ablation Store
    
    負責儲存消融測試 (Ablation Study) 的實驗配置與評估指標。
    使用 PostgreSQL 作為結構化數據的長期存儲。
    """
    
    def __init__(self, db_url: Optional[str] = None):
        if not db_url:
            user = os.getenv("POSTGRES_USER", "saccade")
            password = os.getenv("POSTGRES_PASSWORD", "saccade")
            host = os.getenv("POSTGRES_HOST", "localhost")
            port = os.getenv("POSTGRES_PORT", "5432")
            db = os.getenv("POSTGRES_DB", "saccade")
            # 使用 asyncpg 作為非同步驅動
            db_url = f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db}"
        
        self.engine = create_async_engine(db_url)
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False)

    async def create_tables(self):
        """同步 ORM 模型到資料庫 (通常由 Docker init.sql 處理，此處提供作為手動更新手段)"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def add_experiment(
        self, 
        name: str, 
        category: str, 
        config: Dict[str, Any], 
        description: Optional[str] = None
    ) -> int:
        """紀錄一個新的消融實驗配置"""
        async with self.session_factory() as session:
            exp = Experiment(
                name=name, 
                category=category, 
                config=config, 
                description=description
            )
            session.add(exp)
            await session.commit()
            return exp.id

    async def log_metrics(
        self, 
        experiment_id: int, 
        dataset: str, 
        metrics: Dict[str, Any], 
        raw_results: Optional[Dict[str, Any]] = None
    ):
        """為特定實驗紀錄評估結果"""
        async with self.session_factory() as session:
            metric = Metric(
                experiment_id=experiment_id,
                dataset=dataset,
                idf1=metrics.get("idf1") or metrics.get("IDF1"),
                recall=metrics.get("recall") or metrics.get("Rcll"),
                precision=metrics.get("precision") or metrics.get("Prcn"),
                mota=metrics.get("mota") or metrics.get("MOTA"),
                num_switches=metrics.get("num_switches") or metrics.get("IDs"),
                num_false_positives=metrics.get("num_false_positives") or metrics.get("FP"),
                num_misses=metrics.get("num_misses") or metrics.get("FN"),
                raw_results=raw_results
            )
            session.add(metric)
            await session.commit()

    async def get_experiment_history(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """獲取實驗歷史紀錄與其對應的最佳指標"""
        async with self.session_factory() as session:
            stmt = select(Experiment).order_by(Experiment.created_at.desc())
            if category:
                stmt = stmt.where(Experiment.category == category)
            
            result = await session.execute(stmt)
            experiments = result.scalars().all()
            
            output = []
            for exp in experiments:
                output.append({
                    "id": exp.id,
                    "name": exp.name,
                    "category": exp.category,
                    "config": exp.config,
                    "created_at": exp.created_at.isoformat(),
                    "description": exp.description
                })
            return output

    async def close(self):
        await self.engine.dispose()
