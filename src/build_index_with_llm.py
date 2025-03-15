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


def search_in_topic_tree(tree, query, model=None):
    """在主题树中搜索相关内容"""
    if model is None:
        model = get_embed_model()
    
    query_vector = model.encode(query, normalize_embeddings=True)
    results = []
    
    def search_recursive(node, path=[]):
        current_path = path + [node.title]
        
        # 如果是叶子节点，计算相似度
        if node.is_leaf() and node.content:
            title = node.title
            title_vector = model.encode(title, normalize_embeddings=True)
            similarity = np.dot(query_vector, title_vector)
            
            results.append({
                'title': title,
                'content': node.content,
                'path': current_path,
                'score': float(similarity)
            })
        
        # 递归搜索子节点
        for child in node.children:
            search_recursive(child, current_path)
    
    search_recursive(tree)
    
    # 按相似度排序
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:5]  # 返回前5个最相关的结果


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