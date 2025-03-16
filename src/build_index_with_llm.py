import os
import numpy as np
import json
from tqdm.autonotebook import tqdm
import PyPDF2
from src.embed_model import get_embed_model
from src.build_index_copy import collect_chunks_from_file, collect_chunks_from_dir
import pickle
import time
from openai import OpenAI


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


def call_douban_api(prompt, max_tokens=2000):
    """
    调用豆包API进行主题提取
    
    参数:
        prompt: 提交给模型的提示文本
        max_tokens: 最大生成token数
        
    返回:
        模型生成的响应
    """
    client = OpenAI(
        api_key=os.environ.get("ARK_API_KEY"),
        base_url="https://ark.cn-beijing.volces.com/api/v3",
    )
    
    try:
        response = client.chat.completions.create(
            model="ep-20250216134646-bjpgx",  # 豆包模型ID
            messages=[
                {"role": "system", "content": "你是豆包，是由字节跳动开发的 AI 人工智能助手"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API调用失败: {str(e)}")
        return None


def extract_topics_with_llm(titles, contents, batch_size=20):
    """
    使用豆包LLM提取主题并构建层次结构
    
    参数:
        titles: 文档标题列表
        contents: 文档内容列表
        batch_size: 批处理大小，防止请求过大
        
    返回:
        解析后的主题结构
    """
    all_topics = {"title": "医院综合信息", "children": []}
    
    # 分批处理，避免提示文本过长
    for i in range(0, len(titles), batch_size):
        batch_titles = titles[i:i+batch_size]
        
        # 构建详细的提示文本
        prompt = f"""
我需要你分析以下{len(batch_titles)}个医院文档的标题，并构建一个多层次的主题树结构。
这些文档来自医院不同部门和医疗领域。请根据标题的语义关系，将它们组织成具有层次结构的主题树。

文档标题列表:
{json.dumps(batch_titles, ensure_ascii=False, indent=2)}

请执行以下任务:
1. 分析这些标题，识别主要主题类别和子类别
2. 构建一个层次化的主题树结构，树的顶部是最宏观的主题，从上到下主题的细粒度逐渐增加
3. 每个主题节点应包含一个描述性的名称和与该主题相关的文档索引列表(从1开始编号)
4. 返回JSON格式的树结构，结构如下:
{{
  "title": "根主题名称",
  "children": [
    {{
      "title": "一级主题1",
      "children": [
        {{
          "title": "二级主题1-1",
          "documents": [文档索引列表]
        }},
        ...
      ]
    }},
    ...
  ]
}}

只返回JSON结构，不要有其他解释。确保JSON格式有效，可以被直接解析。
"""
        # 调用豆包API
        response = call_douban_api(prompt)
        if response:
            try:
                # 提取JSON部分
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    batch_structure = json.loads(json_str)
                    
                    # 合并到总主题结构中
                    for category in batch_structure.get("children", []):
                        # 检查是否已存在相同主题
                        existing_category = None
                        for existing in all_topics["children"]:
                            if existing["title"] == category["title"]:
                                existing_category = existing
                                break
                        
                        if existing_category:
                            # 合并子主题
                            for subcategory in category.get("children", []):
                                existing_sub = None
                                for existing in existing_category.get("children", []):
                                    if existing["title"] == subcategory["title"]:
                                        existing_sub = existing
                                        break
                                
                                if existing_sub:
                                    # 合并文档列表
                                    docs = subcategory.get("documents", [])
                                    existing_docs = existing_sub.get("documents", [])
                                    existing_sub["documents"] = list(set(existing_docs + [d+i for d in docs]))
                                else:
                                    # 添加偏移量
                                    if "documents" in subcategory:
                                        subcategory["documents"] = [d+i for d in subcategory["documents"]]
                                    if "children" not in existing_category:
                                        existing_category["children"] = []
                                    existing_category["children"].append(subcategory)
                        else:
                            # 添加偏移量并添加新类别
                            for subcategory in category.get("children", []):
                                if "documents" in subcategory:
                                    subcategory["documents"] = [d+i for d in subcategory["documents"]]
                            all_topics["children"].append(category)
                else:
                    print(f"无法从响应中提取JSON: {response[:100]}...")
            except Exception as e:
                print(f"解析响应时出错: {str(e)}")
                print(f"原始响应: {response[:200]}...")
    
    return all_topics


def build_tree_from_llm_structure(topic_structure, titles, contents):
    """
    将LLM生成的主题结构转换为主题树
    
    参数:
        topic_structure: LLM生成的主题结构
        titles: 文档标题列表
        contents: 文档内容列表
        
    返回:
        构建好的主题树
    """
    def build_node(structure, level=0):
        node = TopicNode(structure["title"])
        
        # 如果有documents字段，将文档添加为叶子节点
        if "documents" in structure:
            for doc_idx in structure["documents"]:
                if 1 <= doc_idx <= len(titles):  # 确保索引有效
                    idx = doc_idx - 1  # 转换为0基索引
                    leaf = TopicNode(titles[idx], contents[idx])
                    node.add_child(leaf)
        
        # 递归处理子节点
        if "children" in structure:
            for child_structure in structure["children"]:
                child_node = build_node(child_structure, level + 1)
                node.add_child(child_node)
        
        return node
    
    # 从根节点开始构建树
    return build_node(topic_structure)


def build_topic_tree_from_dir(dir_path: str):
    """
    从指定目录构建主题树，使用豆包API进行语义理解
    
    参数:
        dir_path: PDF文件所在目录
    
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
    
    print(f"\n★ 开始使用豆包API构建主题树")
    print(f"共有 {len(titles)} 个标题")
    
    # 使用豆包API提取主题
    topic_structure = extract_topics_with_llm(titles, contents)
    
    # 构建主题树
    topic_tree = build_tree_from_llm_structure(topic_structure, titles, contents)
    
    print("主题树构建完成！")
    
    return topic_tree


def search_in_topic_tree(tree, query_vector, model=None, top_branches=2, base_threshold=0.1, depth_factor=0.05):
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
            vector_cache[title] = model.encode(title, normalize_embeddings=True)  # 缓存节点向量
        return vector_cache[title]
    
    def search_recursive(node, path=[], depth=0):
        # 计算动态阈值 - 随深度增加而提高
        current_threshold = base_threshold + depth * depth_factor
        
        current_path = path + [node.title]  # 更新当前路径
        
        # 如果是叶子节点，计算相似度并添加到结果
        if node.is_leaf() and node.content:
            title_vector = get_node_vector(node.title)
            similarity = np.dot(query_vector, title_vector)  # 计算查询向量与节点向量的点积
            
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


def save_topic_tree(tree, filename):
    """保存主题树到文件"""
    with open(filename, 'wb') as f:
        pickle.dump(tree, f)
    print(f"主题树已保存到 {filename}")


def load_topic_tree(filename):
    """从文件加载主题树"""
    with open(filename, 'rb') as f:
        tree = pickle.load(f)
    print(f"已从 {filename} 加载主题树")
    return tree