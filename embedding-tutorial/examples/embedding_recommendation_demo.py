"""
Embedding 推荐系统实战示例

对应小红书教程：推荐系统第 2 课 - Embedding 不用数学公式

这个脚本演示：
1. 如何用 Embedding 表示用户和物品
2. 如何用向量相似度做推荐
3. 如何结合协同过滤 + Embedding

运行方式：
cd /Users/changwang/clawd/financial-recommender/backend
python examples/embedding_recommendation_demo.py
"""
import sys
import os
# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.embedding_engine import EmbeddingEngine, EmbeddingResult, EmbeddingVisualizer
from typing import List, Dict
import json


# ============ 模拟数据 ============

MOVIES = [
    {"id": "m1", "title": "三体", "genre": "科幻", "year": 2023},
    {"id": "m2", "title": "流浪地球", "genre": "科幻", "year": 2019},
    {"id": "m3", "title": "星际穿越", "genre": "科幻", "year": 2014},
    {"id": "m4", "title": "盗梦空间", "genre": "科幻", "year": 2010},
    {"id": "m5", "title": "战狼 2", "genre": "动作", "year": 2017},
    {"id": "m6", "title": "红海行动", "genre": "动作", "year": 2018},
    {"id": "m7", "title": "唐人街探案", "genre": "喜剧", "year": 2015},
    {"id": "m8", "title": "你好，李焕英", "genre": "喜剧", "year": 2021},
]

USERS = [
    {"id": "u1", "name": "小明", "liked": ["三体", "流浪地球", "星际穿越"]},
    {"id": "u2", "name": "小红", "liked": ["战狼 2", "红海行动", "唐人街探案"]},
    {"id": "u3", "name": "小刚", "liked": ["盗梦空间", "星际穿越", "三体"]},
]


class MovieRecommender:
    """
    电影推荐系统
    
    核心思想（协同过滤 + Embedding）：
    1. 找和你品味相似的人（用户 Embedding 相似度）
    2. 找和你喜欢的物品相似的物品（物品 Embedding 相似度）
    3. 两者结合 = 更强的推荐
    """
    
    def __init__(self):
        self.engine = EmbeddingEngine(dimension=256)
        self.movies = MOVIES
        self.users = USERS
        
        # 预计算所有电影的 Embedding
        self.movie_embeddings = {}
        self._precompute_movie_embeddings()
    
    def _precompute_movie_embeddings(self):
        """预计算电影 Embedding"""
        for movie in self.movies:
            # 用标题 + 类型 + 年份生成 Embedding
            text = f"{movie['title']} {movie['genre']} {movie['year']}"
            result = self.engine.generate_embedding(text)
            self.movie_embeddings[movie['id']] = result
    
    def get_user_embedding(self, user: Dict) -> EmbeddingResult:
        """
        生成用户 Embedding
        
        方法：把用户喜欢的所有电影标题拼接起来，生成一个向量
        这个向量代表用户的"品味"
        """
        liked_titles = " ".join(user['liked'])
        return self.engine.generate_embedding(liked_titles)
    
    def find_similar_users(self, target_user: Dict, top_k: int = 2) -> List[Dict]:
        """
        找品味相似的用户（基于用户 Embedding）
        
        这就是"基于用户的协同过滤"
        """
        target_emb = self.get_user_embedding(target_user)
        
        scored_users = []
        for user in self.users:
            if user['id'] == target_user['id']:
                continue
            
            user_emb = self.get_user_embedding(user)
            similarity = self.engine.cosine_similarity(target_emb.embedding, user_emb.embedding)
            scored_users.append((user, similarity))
        
        scored_users.sort(key=lambda x: x[1], reverse=True)
        return scored_users[:top_k]
    
    def recommend_by_content(self, user: Dict, top_k: int = 3) -> List[Dict]:
        """
        基于内容的推荐（Embedding 相似度）
        
        用户喜欢什么，就推荐相似的东西
        """
        # 1. 生成用户偏好向量（喜欢的电影的平均向量）
        user_emb = self.get_user_embedding(user)
        
        # 2. 找相似的电影
        scored_movies = []
        for movie in self.movies:
            # 跳过用户已经看过的
            if movie['title'] in user['liked']:
                continue
            
            movie_emb = self.movie_embeddings[movie['id']]
            similarity = self.engine.cosine_similarity(user_emb.embedding, movie_emb.embedding)
            scored_movies.append((movie, similarity))
        
        # 3. 排序返回
        scored_movies.sort(key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for movie, score in scored_movies[:top_k]:
            recommendations.append({
                "movie": movie,
                "score": score,
                "reason": "与你喜欢的电影相似"
            })
        
        return recommendations
    
    def recommend_by_collaborative(self, user: Dict, top_k: int = 3) -> List[Dict]:
        """
        基于协同过滤的推荐
        
        找品味相似的人，看他们喜欢什么
        """
        # 1. 找相似用户
        similar_users = self.find_similar_users(user, top_k=2)
        
        print(f"\n👥 找到品味相似的用户：")
        for su, score in similar_users:
            print(f"   - {su['name']} (相似度：{score:.3f})")
            print(f"     喜欢：{', '.join(su['liked'])}")
        
        # 2. 收集相似用户喜欢的电影
        candidate_movies = {}
        for su, _ in similar_users:
            for title in su['liked']:
                if title not in user['liked']:  # 排除用户已看过的
                    candidate_movies[title] = candidate_movies.get(title, 0) + 1
        
        # 3. 推荐得票最高的
        recommendations = []
        for title, votes in sorted(candidate_movies.items(), key=lambda x: x[1], reverse=True)[:top_k]:
            movie = next((m for m in self.movies if m['title'] == title), None)
            if movie:
                recommendations.append({
                    "movie": movie,
                    "score": votes / len(similar_users),
                    "reason": f"{votes} 个品味相似的人也喜欢"
                })
        
        return recommendations
    
    def hybrid_recommend(self, user: Dict, top_k: int = 3) -> List[Dict]:
        """
        混合推荐：协同过滤 + Embedding
        
        两者结合 = 更强的推荐系统！
        """
        print("\n" + "=" * 60)
        print(f"🎬 为用户 '{user['name']}' 生成推荐")
        print("=" * 60)
        
        print(f"\n📚 已知偏好：{', '.join(user['liked'])}")
        
        # 1. 基于内容的推荐
        print("\n📌 方法 1：基于内容（Embedding 相似度）")
        content_recs = self.recommend_by_content(user, top_k)
        for i, rec in enumerate(content_recs, 1):
            print(f"   {i}. {rec['movie']['title']} (相似度：{rec['score']:.3f})")
        
        # 2. 协同过滤推荐
        print("\n📌 方法 2：协同过滤（找相似的人）")
        collab_recs = self.recommend_by_collaborative(user, top_k)
        for i, rec in enumerate(collab_recs, 1):
            print(f"   {i}. {rec['movie']['title']} (推荐度：{rec['score']:.3f})")
        
        # 3. 混合排序（简单平均）
        all_recs = {}
        for rec in content_recs + collab_recs:
            title = rec['movie']['title']
            if title not in all_recs:
                all_recs[title] = {"movie": rec['movie'], "content_score": 0, "collab_score": 0}
            
            if rec['reason'] == "与你喜欢的电影相似":
                all_recs[title]['content_score'] = rec['score']
            else:
                all_recs[title]['collab_score'] = rec['score']
        
        # 计算综合分数
        hybrid_recs = []
        for title, data in all_recs.items():
            hybrid_score = (data['content_score'] + data['collab_score']) / 2
            hybrid_recs.append({
                "movie": data['movie'],
                "score": hybrid_score,
                "content_score": data['content_score'],
                "collab_score": data['collab_score']
            })
        
        hybrid_recs.sort(key=lambda x: x['score'], reverse=True)
        
        print("\n🏆 最终推荐（混合排序）：")
        for i, rec in enumerate(hybrid_recs[:top_k], 1):
            print(f"   {i}. {rec['movie']['title']}")
            print(f"      综合分：{rec['score']:.3f}")
            print(f"      (内容：{rec['content_score']:.3f}, 协同：{rec['collab_score']:.3f})")
        
        return hybrid_recs[:top_k]


def demo_embedding_visualization():
    """演示 Embedding 可视化"""
    print("\n" + "=" * 60)
    print("📊 Embedding 可视化演示")
    print("=" * 60)
    
    engine = EmbeddingEngine(dimension=64)
    visualizer = EmbeddingVisualizer()
    
    # 添加电影 Embedding
    for movie in MOVIES:
        text = f"{movie['title']} {movie['genre']}"
        result = engine.generate_embedding(text)
        visualizer.add(result.embedding, f"{movie['title']}({movie['genre']})")
    
    # 保存可视化图
    output_path = "/Users/changwang/Downloads/电影 Embedding 可视化.png"
    visualizer.plot_2d(save_path=output_path)
    
    print(f"\n✅ 可视化完成！")
    print(f"查看：{output_path}")
    print("\n你会看到：")
    print("  - 科幻电影聚在一起（三体、流浪地球、星际穿越...）")
    print("  - 动作电影聚在一起（战狼 2、红海行动...）")
    print("  - 喜剧电影聚在一起（唐人街探案、你好李焕英...）")
    print("\n这就是'相似的东西距离近'！")


def main():
    """主演示"""
    print("=" * 60)
    print("🎬 Embedding 推荐系统实战演示")
    print("=" * 60)
    print("\n对应小红书教程：推荐系统第 2 课 - Embedding")
    print("文案位置：/Users/changwang/Downloads/自媒体/2026-04-08-Embedding 解释.md")
    
    # 创建推荐系统
    recommender = MovieRecommender()
    
    # 为新用户推荐（假设他喜欢科幻）
    new_user = {
        "id": "u_new",
        "name": "新用户",
        "liked": ["三体", "流浪地球"]  # 刚看了这两部科幻片
    }
    
    # 生成混合推荐
    recommendations = recommender.hybrid_recommend(new_user, top_k=3)
    
    # 可视化演示
    demo_embedding_visualization()
    
    print("\n" + "=" * 60)
    print("✅ 演示完成！")
    print("=" * 60)
    
    print("\n📚 关键知识点回顾：")
    print("1. Embedding = 把东西变成数字，让电脑能理解")
    print("2. 相似的东西，数字也相似（向量距离近）")
    print("3. 应用：搜索、推荐、分类")
    print("4. 协同过滤 + Embedding = 更强的推荐系统")
    
    print("\n💡 下一步：")
    print("  - 查看代码：/Users/changwang/clawd/financial-recommender/backend/utils/embedding_engine.py")
    print("  - 运行示例：python examples/embedding_recommendation_demo.py")
    print("  - 小红书文案：/Users/changwang/Downloads/自媒体/2026-04-08-Embedding 解释.md")


if __name__ == "__main__":
    main()
