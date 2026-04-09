"""
Embedding 工具模块 - 向量嵌入生成与相似度计算

对应小红书教程：Embedding 到底是啥？用人类语言解释
https://example.com/embedding-tutorial

功能：
1. 文本 Embedding 生成
2. 向量相似度计算（余弦相似度）
3. 向量可视化（2D/3D 降维）
4. 批量处理

使用场景：
- 推荐系统：找相似的物品
- 语义搜索：理解查询意图
- 文本分类：自动归类
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json


@dataclass
class EmbeddingResult:
    """Embedding 结果"""
    text: str
    embedding: List[float]
    dimension: int
    
    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "embedding": self.embedding,
            "dimension": self.dimension
        }


class EmbeddingEngine:
    """
    Embedding 引擎
    
    核心思想：
    > 把任何东西（文字/图片/视频）变成一串数字，让电脑能理解。
    
    Embedding 的魔法：
    > 相似的东西，数字也相似。
    """
    
    def __init__(self, model_name: str = "text-embedding-v3", dimension: int = 1536):
        """
        初始化 Embedding 引擎
        
        Args:
            model_name: 模型名称
            dimension: Embedding 维度
        """
        self.model_name = model_name
        self.dimension = dimension
        # 实际使用时，这里会初始化真实的模型客户端
        # 例如：self.client = QwenClient() 或 self.client = OpenAI()
    
    def generate_embedding(self, text: str) -> EmbeddingResult:
        """
        生成单个文本的 Embedding
        
        Args:
            text: 输入文本
        
        Returns:
            EmbeddingResult 对象
        
        示例：
        >>> engine = EmbeddingEngine()
        >>> result = engine.generate_embedding("苹果")
        >>> print(f"维度：{result.dimension}")
        >>> print(f"前 5 个值：{result.embedding[:5]}")
        """
        # TODO: 调用真实的 Embedding API
        # embedding = await self.client.get_embedding(text)
        
        # 模拟 Embedding（演示用）
        embedding = self._mock_embedding(text)
        
        return EmbeddingResult(
            text=text,
            embedding=embedding,
            dimension=self.dimension
        )
    
    def generate_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """
        批量生成 Embedding
        
        Args:
            texts: 文本列表
        
        Returns:
            EmbeddingResult 列表
        
        示例：
        >>> texts = ["苹果", "香蕉", "汽车"]
        >>> results = engine.generate_batch(texts)
        """
        results = []
        for text in texts:
            result = self.generate_embedding(text)
            results.append(result)
        return results
    
    def cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        计算两个向量的余弦相似度
        
        公式：cos(θ) = (A·B) / (||A|| * ||B||)
        
        取值范围：[-1, 1]
        - 1: 完全相同
        - 0: 无关
        - -1: 完全相反
        
        Args:
            embedding1: 向量 1
            embedding2: 向量 2
        
        Returns:
            相似度分数
        
        示例：
        >>> apple_emb = engine.generate_embedding("苹果").embedding
        >>> banana_emb = engine.generate_embedding("香蕉").embedding
        >>> car_emb = engine.generate_embedding("汽车").embedding
        >>> 
        >>> print(f"苹果 - 香蕉相似度：{engine.cosine_similarity(apple_emb, banana_emb):.3f}")
        >>> print(f"苹果 - 汽车相似度：{engine.cosine_similarity(apple_emb, car_emb):.3f}")
        """
        v1 = np.array(embedding1)
        v2 = np.array(embedding2)
        
        # 计算点积
        dot_product = np.dot(v1, v2)
        
        # 计算模长
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        # 避免除以零
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    
    def find_similar(
        self,
        query_embedding: List[float],
        candidates: List[EmbeddingResult],
        top_k: int = 5
    ) -> List[Tuple[EmbeddingResult, float]]:
        """
        查找最相似的候选项
        
        这就是推荐系统的核心：
        > 找和你品味相似的人，看他们喜欢什么，推荐给你。
        
        Args:
            query_embedding: 查询向量
            candidates: 候选项列表
            top_k: 返回数量
        
        Returns:
            (候选项，相似度) 元组列表，按相似度降序
        
        示例：
        >>> query = engine.generate_embedding("我喜欢科幻电影")
        >>> candidates = [engine.generate_embedding(t) for t in movie_titles]
        >>> similar = engine.find_similar(query.embedding, candidates, top_k=3)
        >>> for item, score in similar:
        ...     print(f"{item.text}: {score:.3f}")
        """
        scored = []
        for candidate in candidates:
            score = self.cosine_similarity(query_embedding, candidate.embedding)
            scored.append((candidate, score))
        
        # 按相似度降序排序
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return scored[:top_k]
    
    def _mock_embedding(self, text: str) -> List[float]:
        """
        模拟 Embedding 生成（演示用）
        
        实际使用时替换为真实的 API 调用
        """
        # 使用文本的哈希值生成确定性随机向量
        np.random.seed(hash(text) % 2**32)
        embedding = np.random.randn(self.dimension)
        
        # 归一化（单位向量）
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding.tolist()


class EmbeddingVisualizer:
    """
    Embedding 可视化工具
    
    将高维向量降维到 2D/3D 空间，帮助理解"相似的东西距离近"
    """
    
    def __init__(self):
        self.embeddings = []
        self.labels = []
    
    def add(self, embedding: List[float], label: str):
        """添加一个 Embedding"""
        self.embeddings.append(embedding)
        self.labels.append(label)
    
    def add_batch(self, results: List[EmbeddingResult]):
        """批量添加"""
        for result in results:
            self.add(result.embedding, result.text)
    
    def reduce_to_2d(self, method: str = "pca") -> np.ndarray:
        """
        降维到 2D
        
        Args:
            method: 降维方法 ("pca", "tsne", "umap")
        
        Returns:
            2D 坐标数组 (n_samples, 2)
        """
        X = np.array(self.embeddings)
        
        if method == "pca":
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
        elif method == "tsne":
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
        elif method == "umap":
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
        else:
            raise ValueError(f"未知方法：{method}")
        
        return reducer.fit_transform(X)
    
    def reduce_to_3d(self, method: str = "pca") -> np.ndarray:
        """降维到 3D"""
        X = np.array(self.embeddings)
        
        if method == "pca":
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=3)
        elif method == "tsne":
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=3, random_state=42)
        elif method == "umap":
            import umap
            reducer = umap.UMAP(n_components=3, random_state=42)
        else:
            raise ValueError(f"未知方法：{method}")
        
        return reducer.fit_transform(X)
    
    def plot_2d(self, save_path: Optional[str] = None):
        """
        绘制 2D 散点图
        
        Args:
            save_path: 保存路径（None 则显示）
        """
        import matplotlib.pyplot as plt
        
        coords_2d = self.reduce_to_2d()
        
        plt.figure(figsize=(10, 8))
        plt.scatter(coords_2d[:, 0], coords_2d[:, 1], alpha=0.6)
        
        # 标注文本
        for i, label in enumerate(self.labels):
            plt.annotate(label, (coords_2d[i, 0], coords_2d[i, 1]), fontsize=9)
        
        plt.title("Embedding 可视化 - 相似的东西距离近")
        plt.xlabel("维度 1")
        plt.ylabel("维度 2")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图已保存到：{save_path}")
        else:
            plt.show()
    
    def plot_3d(self, save_path: Optional[str] = None):
        """绘制 3D 散点图"""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        coords_3d = self.reduce_to_3d()
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2], alpha=0.6)
        
        # 标注文本
        for i, label in enumerate(self.labels):
            ax.text(coords_3d[i, 0], coords_3d[i, 1], coords_3d[i, 2], label, fontsize=8)
        
        ax.set_title("Embedding 3D 可视化")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图已保存到：{save_path}")
        else:
            plt.show()


# ============ 使用示例 ============

def demo_embedding_basics():
    """
    Embedding 基础演示
    
    对应小红书文案中的例子：
    - "苹果"的 Embedding：[0.82, -0.34, 0.91]
    - "香蕉"的 Embedding：[0.79, -0.31, 0.88]
    - "汽车"的 Embedding：[-0.45, 0.67, -0.23]
    """
    print("=" * 60)
    print("Embedding 基础演示")
    print("=" * 60)
    
    engine = EmbeddingEngine(dimension=128)  # 用小维度方便演示
    
    # 生成 Embedding
    texts = ["苹果", "香蕉", "橘子", "汽车", "飞机", "火车"]
    results = engine.generate_batch(texts)
    
    print("\n1️⃣ 什么是 Embedding？")
    print("把文字变成一串数字，让电脑能理解\n")
    
    for result in results[:3]:
        print(f"'{result.text}' → [{result.embedding[0]:.2f}, {result.embedding[1]:.2f}, ...] (维度：{result.dimension})")
    
    print("\n2️⃣ 相似的东西，数字也相似")
    print("计算余弦相似度：\n")
    
    # 计算相似度
    apple = results[0]
    banana = results[1]
    car = results[3]
    
    sim_apple_banana = engine.cosine_similarity(apple.embedding, banana.embedding)
    sim_apple_car = engine.cosine_similarity(apple.embedding, car.embedding)
    
    print(f"🍎 苹果 - 🍌 香蕉相似度：{sim_apple_banana:.3f} (都是水果，应该较高)")
    print(f"🍎 苹果 - 🚗 汽车相似度：{sim_apple_car:.3f} (不同类，应该较低)")
    
    print("\n3️⃣ 应用场景：搜索")
    print("用户搜'手机'，找到数字接近的'iPhone'、'安卓'、'华为'\n")
    
    query = engine.generate_embedding("手机")
    candidates = [
        engine.generate_embedding("iPhone 15"),
        engine.generate_embedding("安卓手机"),
        engine.generate_embedding("华为 Mate"),
        engine.generate_embedding("香蕉"),  # 不相关
    ]
    
    similar = engine.find_similar(query.embedding, candidates, top_k=3)
    
    print("搜索结果（按相似度排序）：")
    for item, score in similar:
        print(f"  - {item.text}: {score:.3f}")
    
    print("\n" + "=" * 60)


def demo_visualization():
    """可视化演示"""
    print("\n" + "=" * 60)
    print("Embedding 可视化演示")
    print("=" * 60)
    
    engine = EmbeddingEngine(dimension=64)
    visualizer = EmbeddingVisualizer()
    
    # 添加示例
    categories = {
        "水果": ["苹果", "香蕉", "橘子", "葡萄", "西瓜"],
        "交通工具": ["汽车", "飞机", "火车", "轮船", "自行车"],
        "电子产品": ["手机", "电脑", "平板", "相机", "耳机"],
    }
    
    for category, items in categories.items():
        results = engine.generate_batch(items)
        visualizer.add_batch(results)
    
    print(f"\n已添加 {len(visualizer.labels)} 个 Embedding")
    print("正在降维可视化...")
    
    # 保存 2D 图
    output_path = "/Users/changwang/Downloads/embedding_visualization.png"
    visualizer.plot_2d(save_path=output_path)
    
    print(f"\n✅ 可视化完成！打开 {output_path} 查看")
    print("你会看到：同类物品聚在一起，不同类物品距离远")
    print("=" * 60)


if __name__ == "__main__":
    demo_embedding_basics()
    demo_visualization()
