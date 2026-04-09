# Embedding 配套代码

📚 **对应小红书教程：** [Embedding 到底是啥？用人类语言解释]
---

## 📁 文件结构

```
embedding-tutorial/
├── README.md                    # 快速开始
├── utils/
│   └── embedding_engine.py      # Embedding 引擎核心代码
├── examples/
│   └── embedding_recommendation_demo.py  # 完整示例
├── tests/
│   └── test_embedding.py        # 单元测试
└── docs/
    └── EMBEDDING_GUIDE.md       # 本文件
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
cd /Users/changwang/xiaohongshu-codes/embedding-tutorial
pip install numpy scikit-learn matplotlib
```

### 2. 运行演示

```bash
# 运行完整示例
python examples/embedding_recommendation_demo.py

# 运行测试
python -m pytest tests/test_embedding.py -v
```

### 3. 代码示例

```python
from utils.embedding_engine import EmbeddingEngine

# 创建引擎
engine = EmbeddingEngine(dimension=1536)

# 生成 Embedding
result = engine.generate_embedding("苹果")
print(f"维度：{result.dimension}")
print(f"前 5 个值：{result.embedding[:5]}")

# 计算相似度
apple = engine.generate_embedding("苹果")
banana = engine.generate_embedding("香蕉")
car = engine.generate_embedding("汽车")

sim_fruit = engine.cosine_similarity(apple.embedding, banana.embedding)
sim_cross = engine.cosine_similarity(apple.embedding, car.embedding)

print(f"苹果 - 香蕉：{sim_fruit:.3f}")
print(f"苹果 - 汽车：{sim_cross:.3f}")
```

---

## 📖 核心概念

### 什么是 Embedding？

> 把任何东西（文字/图片/视频）变成一串数字，让电脑能理解。

**为什么需要？**
- 电脑看不懂"苹果"这个词
- 但能看懂 `[0.82, -0.34, 0.91, ...]` 这串数字

### Embedding 的魔法

> 相似的东西，数字也相似。

```
"苹果" → [0.82, -0.34, 0.91]
"香蕉" → [0.79, -0.31, 0.88]  ← 数字接近（都是水果）
"汽车" → [-0.45, 0.67, -0.23] ← 数字差得远（不是同类）
```

### 应用场景

| 场景 | 说明 | 代码示例 |
|------|------|---------|
| **搜索** | 搜"手机"，找到"iPhone""安卓" | `find_similar()` |
| **推荐** | 喜欢《三体》，推荐《流浪地球》 | `recommend_by_content()` |
| **分类** | 自动归类新闻、邮件 | 聚类算法 |

---

## 🔧 API 参考

### EmbeddingEngine

#### `generate_embedding(text: str) -> EmbeddingResult`
生成单个文本的 Embedding

```python
result = engine.generate_embedding("测试文本")
print(result.embedding)  # 向量
print(result.dimension)  # 维度
```

#### `generate_batch(texts: List[str]) -> List[EmbeddingResult]`
批量生成

```python
results = engine.generate_batch(["苹果", "香蕉", "汽车"])
```

#### `cosine_similarity(emb1, emb2) -> float`
计算余弦相似度

```python
sim = engine.cosine_similarity(emb1.embedding, emb2.embedding)
# 范围：[-1, 1]，越接近 1 越相似
```

#### `find_similar(query_emb, candidates, top_k) -> List[Tuple[EmbeddingResult, float]]`
查找最相似的候选项

```python
similar = engine.find_similar(query.embedding, candidates, top_k=5)
for item, score in similar:
    print(f"{item.text}: {score:.3f}")
```

### EmbeddingVisualizer

#### `plot_2d(save_path=None)`
绘制 2D 散点图

```python
viz = EmbeddingVisualizer()
viz.add_batch(results)
viz.plot_2d(save_path="visualization.png")
```

---

## 🎯 推荐系统实战

### 协同过滤 + Embedding

```python
from examples.embedding_recommendation_demo import MovieRecommender

recommender = MovieRecommender()

user = {
    "id": "u1",
    "name": "小明",
    "liked": ["三体", "流浪地球", "星际穿越"]
}

# 混合推荐（协同过滤 + Embedding）
recommendations = recommender.hybrid_recommend(user, top_k=3)

for rec in recommendations:
    print(f"{rec['movie']['title']}: {rec['score']:.3f}")
```

### 输出示例

```
🎬 为用户 '小明' 生成推荐
============================================================

📚 已知偏好：三体，流浪地球，星际穿越

📌 方法 1：基于内容（Embedding 相似度）
   1. 盗梦空间 (相似度：0.876)
   2. 星际穿越 2 (相似度：0.823)

📌 方法 2：协同过滤（找相似的人）
👥 找到品味相似的用户：
   - 小刚 (相似度：0.912)
     喜欢：盗梦空间，星际穿越，三体
   1. 盗梦空间 (推荐度：0.500)

🏆 最终推荐（混合排序）：
   1. 盗梦空间
      综合分：0.688
      (内容：0.876, 协同：0.500)
```

---

## 📊 可视化示例

运行演示后，会在 `/Users/changwang/Downloads/` 生成：

1. **电影 Embedding 可视化.png** - 展示电影在 2D 空间的分布
   - 科幻电影聚在一起
   - 动作电影聚在一起
   - 喜剧电影聚在一起

2. **embedding_visualization.png** - 通用 Embedding 可视化

---

## 🧪 测试

```bash
# 运行所有测试
python -m pytest tests/test_embedding.py -v

# 运行特定测试
python -m pytest tests/test_embedding.py::TestEmbeddingEngine::test_cosine_similarity -v

# 查看覆盖率
python -m pytest tests/test_embedding.py --cov=utils/embedding_engine
```

---

## 🔗 与现有代码集成

### 在 RecommendationAgent 中使用

现有代码 (`cognition/recommendation_agent.py`) 已经使用了 Embedding：

```python
# 生成查询文本
query_text = self._build_query_text(context)

# 获取 Embedding
query_embedding = await self.qwen_client.get_embedding(query_text)

# 向量搜索
candidates = await self._search_candidates(query_embedding, filters)
```

### 替换为本地 Embedding 引擎

```python
from utils.embedding_engine import EmbeddingEngine

class RecommendationAgent:
    def __init__(self):
        self.embedding_engine = EmbeddingEngine(dimension=1536)
    
    async def recommend(self, user_id: str, context: Dict):
        # 生成本地 Embedding
        query_text = self._build_query_text(context)
        result = self.embedding_engine.generate_embedding(query_text)
        
        # 继续后续流程...
```

---

## 💡 常见问题

### Q: 为什么用余弦相似度？
A: 余弦相似度衡量方向而非大小，适合比较语义相似性。两个向量方向越接近，内容越相似。

### Q: Embedding 维度怎么选？
A: 
- 128-256: 轻量级应用，速度快
- 512-1024: 平衡性能和精度
- 1536+: 高精度场景（如专业领域）

### Q: 如何提升 Embedding 质量？
A:
1. 使用更大的预训练模型
2. 在领域数据上微调（Fine-tuning）
3. 用更好的文本表示（标题 + 描述 + 标签）

---



## 🎯 下一步

1. **运行示例**：`python examples/embedding_recommendation_demo.py`
2. **修改代码**：尝试添加自己的数据
3. **发布笔记**：用文案模板创作小红书内容

---

**创建日期：** 2026-04-08  
**对应教程：** 推荐系统入门系列 第 2 期
��：** 推荐系统入门系列 第 2 期
