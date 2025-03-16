import os
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
from lab_1806_vec_db import VecDB
from tqdm.autonotebook import tqdm
import PyPDF2
from src.embed_model import get_embed_model
from src.build_index_copy import collect_chunks_from_file,collect_chunks_from_dir
import pickle


class TopicNode:
    """主题树节点"""
    def __init__(self, title=None, content=None):
        self.title = title  # 节点标题
        self.content = content  # 文本内容（叶子节点）
        self.children = []  # 子节点
        self.level = 0  # 节点层级
        self.vector = None  # 标题的向量表示
    
    def add_child(self, child):
        """添加子节点"""
        child.level = self.level + 1
        self.children.append(child)
    
    def is_leaf(self):
        """判断是否为叶子节点"""
        return len(self.children) == 0
    
    def __str__(self):
        """打印节点信息"""
        prefix = "  " * self.level
        result = f"{prefix}{self.title}"
        if self.is_leaf() and self.content:
            result += f" (内容长度: {len(self.content)} 字符)"
        return result
    
    def print_tree(self):
        """打印整棵树"""
        print(self)
        for child in self.children:
            child.print_tree()


def build_topic_tree_from_dir(dir_path: str, max_depth=3, min_cluster_size=2):
    """
    从指定目录构建主题树，不涉及数据库操作
    
    参数:
        dir_path: PDF文件所在目录
        max_depth: 树的最大深度
        min_cluster_size: 最小聚类大小
    
    返回:
        topic_tree: 构建好的主题树
    """
    # 收集所有PDF文件的内容
    chunks = collect_chunks_from_dir(dir_path)
    
    # 提取标题和内容
    titles = []
    contents = []
    for chunk in chunks:
        # 从文本块中提取标题（去除"# "前缀）
        title = chunk.split("\n")[0].replace('# ', '')
        titles.append(title)
        contents.append(chunk)
    
    print(f"\n★ 开始构建主题树")
    print(f"共有 {len(titles)} 个标题")
    
    # 使用BERT模型获取标题的向量表示
    model = get_embed_model()
    title_vectors = model.encode(titles, normalize_embeddings=True)
    
    # 构建主题树
    topic_tree = create_topic_hierarchy(titles, contents, title_vectors, max_depth, min_cluster_size)
    
    print("主题树构建完成！")
    
    return topic_tree


def create_topic_hierarchy(titles, contents, vectors, max_depth=3, min_cluster_size=2):
    """
    创建主题层次结构
    
    参数:
        titles: 文档标题列表
        contents: 文档内容列表
        vectors: 标题的向量表示
        max_depth: 树的最大深度
        min_cluster_size: 最小聚类大小
    
    返回:
        root: 主题树的根节点
    """
    # 创建根节点
    root = TopicNode("全部文档")
    
    # 递归构建主题树
    build_tree_recursive(root, titles, contents, vectors, 1, max_depth, min_cluster_size)
    
    return root


def build_tree_recursive(parent_node, titles, contents, vectors, current_depth, max_depth, min_cluster_size):
    """递归构建主题树"""
    n_samples = len(titles)
    
    # 如果样本数量少于最小聚类大小或已达到最大深度，则直接作为叶子节点
    if n_samples <= min_cluster_size or current_depth >= max_depth:
        for title, content in zip(titles, contents):
            leaf = TopicNode(title, content)
            parent_node.add_child(leaf)
        return
    
    # 使用层次聚类算法
    if n_samples > 2:
        # 尝试将标题聚类成多个组
        n_clusters = min(max(2, n_samples // 3), 5)  # 动态确定聚类数量
        clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='cosine', linkage='average')
        cluster_labels = clustering.fit_predict(vectors)
        
        # 找出每个聚类的中心样本（最接近聚类中心的样本）
        cluster_centers = {}
        for i in range(n_clusters):
            cluster_indices = np.where(cluster_labels == i)[0]
            if len(cluster_indices) == 0:
                continue
                
            # 计算聚类中心
            cluster_center_vector = np.mean(vectors[cluster_indices], axis=0)
            cluster_center_vector /= np.linalg.norm(cluster_center_vector)
            
            # 找到最接近中心的样本
            similarities = np.dot(vectors[cluster_indices], cluster_center_vector)
            center_idx = cluster_indices[np.argmax(similarities)]
            
            cluster_centers[i] = titles[center_idx]
            
            # 创建聚类节点
            cluster_node = TopicNode(titles[center_idx])
            parent_node.add_child(cluster_node)
            
            # 递归构建子树
            sub_titles = [titles[j] for j in cluster_indices]
            sub_contents = [contents[j] for j in cluster_indices]
            sub_vectors = vectors[cluster_indices]
            
            build_tree_recursive(
                cluster_node, sub_titles, sub_contents, sub_vectors, 
                current_depth + 1, max_depth, min_cluster_size
            )
    else:
        # 样本太少，直接作为叶子节点
        for title, content in zip(titles, contents):
            leaf = TopicNode(title, content)
            parent_node.add_child(leaf)


def save_topic_tree(tree, file_path):
    """将主题树保存到文件"""
    with open(file_path, 'wb') as f:
        pickle.dump(tree, f)


def load_topic_tree(file_path):
    """从文件加载主题树"""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def search_in_topic_tree(tree, query_vector, model=None, top_branches=2, base_threshold=0.2, depth_factor=0.1):
    """
    利用树的层次结构进行高效检索，使用动态阈值
    
    参数:
        tree: 主题树
        query: 查询文本
        model: 嵌入模型
        top_branches: 每层选择的最相关分支数量
        base_threshold: 基础相似度阈值
        depth_factor: 深度影响因子，控制阈值随深度增加的速率
    
    返回:
        按相关度排序的结果列表
    """
    if model is None:
        model = get_embed_model()
    # 存储所有找到的结果
    all_results = []
    
    # 缓存节点向量，避免重复计算
    vector_cache = {}
    
    def get_node_vector(title):
        """获取节点标题的向量表示，带缓存"""
        if title not in vector_cache:
            vector_cache[title] = model.encode(title, normalize_embeddings=True)
        return vector_cache[title]
    
    def search_recursive(node, path=[], depth=0):
        # 计算动态阈值 - 随深度增加而提高
        current_threshold = base_threshold + depth * depth_factor
        
        current_path = path + [node.title]
        
        # 如果是叶子节点，计算相似度并添加到结果
        if node.is_leaf() and node.content:
            title_vector = get_node_vector(node.title)
            similarity = np.dot(query_vector, title_vector)
            
            # 叶子节点使用较高阈值
            leaf_threshold = max(current_threshold, 0.4)  # 确保叶子节点至少有0.4的相似度
            
            if similarity >= leaf_threshold:
                all_results.append({
                    'title': node.title,
                    'content': node.content,
                    'path': current_path,
                    'score': float(similarity)
                })
            return
        
        # 对于非叶子节点，计算所有子节点与查询的相似度
        child_similarities = []
        for child in node.children:
            child_vector = get_node_vector(child.title)
            similarity = np.dot(query_vector, child_vector)
            child_similarities.append((child, similarity))
        
        # 按相似度降序排序子节点
        child_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 只遍历相似度最高的前top_branches个分支，且相似度必须高于当前层的阈值
        selected_branches = 0
        for child, similarity in child_similarities:
            if similarity >= current_threshold and selected_branches < top_branches:
                search_recursive(child, current_path, depth + 1)
                selected_branches += 1
            
            # 即使低于阈值但非常接近顶部结果，也考虑探索（避免边界情况）
            elif (selected_branches == 0 and 
                  child_similarities and 
                  similarity >= current_threshold * 0.8):  # 降低20%的阈值作为容错
                search_recursive(child, current_path, depth + 1)
                selected_branches += 1
    
    # 从根节点开始搜索
    search_recursive(tree)
    
    # 结果按相似度排序
    all_results.sort(key=lambda x: x['score'], reverse=True)
    
    # 返回前5个最相关结果，每个结果包装在列表中
    nested_results = [[item] for item in all_results[:5]]
    return nested_results

# 修改收集PDF文件内容的函数，添加正确的标题格式
def collect_chunks_from_pdf(file_path: str):
    """处理单个PDF文件，将整个PDF内容作为一个文本块"""
    chunks = []
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            file_name = os.path.basename(file_path)
            # 去掉.pdf后缀作为标题
            title = os.path.splitext(file_name)[0]
            text = f"# {title}\n\n"
            
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            
            if text.strip():
                chunks.append(text)
                print(f"已提取PDF: {file_name}，内容长度: {len(text)} 字符")
            else:
                print(f"警告: PDF文件 {file_name} 未提取到文本内容")
    except Exception as e:
        print(f"处理PDF文件 {file_path} 时出错: {str(e)}")
    
    return chunks


# 现有函数保持不变，只修改相关部分
def collect_chunks(dir_or_file: str):
    if os.path.isdir(dir_or_file):
        return collect_chunks_from_dir(dir_or_file)
    else:
        return collect_chunks_from_file(dir_or_file)