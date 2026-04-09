# Embedding 教程配套代码

📚 **对应小红书教程：** 推荐系统第 2 课 - Embedding 解释  
📝 **文案位置：** `/Users/changwang/Downloads/自媒体/2026-04-08-Embedding 解释.md`

---

## 📁 项目结构

```
embedding-tutorial/
├── README.md                    # 本文件
├── utils/
│   └── embedding_engine.py      # Embedding 引擎核心代码
├── examples/
│   └── embedding_recommendation_demo.py  # 完整示例
├── tests/
│   └── test_embedding.py        # 单元测试
└── docs/
    └── EMBEDDING_GUIDE.md       # 详细使用文档
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
cd /Users/changwang/xiaohongshu-codes/embedding-tutorial
pip install -r requirements.txt
```

或者手动安装：
```bash
pip install numpy scikit-learn matplotlib pytest
```

### 2. 运行演示

```bash
python examples/embedding_recommendation_demo.py
```

### 3. 运行测试

```bash
python -m pytest tests/test_embedding.py -v
```

---

## 📖 核心概念

### 什么是 Embedding？

> 把任何东西（文字/图片/视频）变成一串数字，让电脑能理解。

### Embedding 的魔法

> 相似的东西，数字也相似。

```
"苹果" → [0.82, -0.34, 0.91]
"香蕉" → [0.79, -0.31, 0.88]  ← 数字接近（都是水果）
"汽车" → [-0.45, 0.67, -0.23] ← 数字差得远（不是同类）
```

### 应用场景

| 场景 | 说明 |
|------|------|
| **搜索** | 搜"手机"，找到"iPhone""安卓" |
| **推荐** | 喜欢《三体》，推荐《流浪地球》 |
| **分类** | 自动归类新闻、邮件 |

---

## 💻 代码示例

### 基础用法

```python
from utils.embedding_engine import EmbeddingEngine

# 创建引擎
engine = EmbeddingEngine(dimension=1536)

# 生成 Embedding
result = engine.generate_embedding("苹果")
print(f"维度：{result.dimension}")

# 计算相似度
apple = engine.generate_embedding("苹果")
banana = engine.generate_embedding("香蕉")
car = engine.generate_embedding("汽车")

sim_fruit = engine.cosine_similarity(apple.embedding, banana.embedding)
sim_cross = engine.cosine_similarity(apple.embedding, car.embedding)

print(f"苹果 - 香蕉：{sim_fruit:.3f}")
print(f"苹果 - 汽车：{sim_cross:.3f}")
```

### 推荐系统示例

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

---

## 📊 可视化

运行演示后，会在 `/Users/changwang/Downloads/` 生成可视化图表：

- **电影 Embedding 可视化.png** - 展示电影在 2D 空间的分布
  - 科幻电影聚在一起
  - 动作电影聚在一起
  - 喜剧电影聚在一起

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

## 📝 输出示例

```
============================================================
🎬 Embedding 推荐系统实战演示
============================================================

🎬 为用户 '新用户' 生成推荐
📚 已知偏好：三体，流浪地球

📌 方法 1：基于内容（Embedding 相似度）
   1. 盗梦空间 (相似度：0.129)
   2. 战狼 2 (相似度：0.050)

📌 方法 2：协同过滤（找相似的人）
👥 找到品味相似的用户：
   - 小刚 (相似度：0.102)

🏆 最终推荐（混合排序）：
   1. 盗梦空间
      综合分：0.314
```

---

## 🔗 相关链接

- **详细文档：** `docs/EMBEDDING_GUIDE.md`
- **小红书文案：** `/Users/changwang/Downloads/自媒体/2026-04-08-Embedding 解释.md`
- **协同过滤（第 1 期）：** `../collaborative_filtering/`
- **可视化输出：** `/Users/changwang/Downloads/电影 Embedding 可视化.png`

---

## 📈 系列教程

| 期数 | 主题 | 状态 |
|------|------|------|
| 第 1 期 | 协同过滤 | ✅ 已发布 |
| 第 2 期 | Embedding | ✅ 代码完成 |
| 第 3 期 | 混合推荐 | 📅 计划中 |

---

## 💡 常见问题

### Q: 为什么用余弦相似度？
A: 余弦相似度衡量方向而非大小，适合比较语义相似性。

### Q: Embedding 维度怎么选？
A: 
- 128-256: 轻量级应用
- 512-1024: 平衡性能和精度
- 1536+: 高精度场景

### Q: 如何提升 Embedding 质量？
A:
1. 使用更大的预训练模型
2. 在领域数据上微调
3. 用更好的文本表示

---

**创建日期：** 2026-04-08  
**对应教程：** 推荐系统入门系列 第 2 期  
**代码状态：** ✅ 完成并测试通过
