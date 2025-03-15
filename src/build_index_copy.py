import os

from lab_1806_vec_db import VecDB
from tqdm.autonotebook import tqdm
import PyPDF2  # 添加这行导入
from src.embed_model import get_embed_model


def split_string_by_headings(text: str):
    print("\n=== 开始文本分块 ===")  # 阶段开始标记
    lines = text.split("\n")
    current_block: list[str] = []
    chunks: list[str] = []

    def concat_block():
        if len(current_block) > 0:
            chunks.append("\n".join(current_block))
            current_block.clear()

    for line in lines:
        if line.startswith("# "):
            concat_block()
        current_block.append(line)
    concat_block()
    print(f"生成 {len(chunks)} 个文本块")
    for i, chunk in enumerate(chunks[:3]):  # 只显示前3个块
       summary = chunk.split("\n")[0][:50]  # 先提取摘要
       print(f'块 {i+1} 摘要：{summary}...')  # 再使用变量
    return chunks


def collect_chunks_from_file(file_path: str):
    with open(file_path, "r", encoding="utf-8") as file:
        data = file.read()
        return split_string_by_headings(data)


def collect_chunks_from_dir(dir: str):
    """从目录中收集PDF文件内容，每个PDF作为一个文本块"""
    chunks: list[str] = []
    for filename in os.listdir(dir):
        if filename.endswith(".pdf"):
            file_path = os.path.join(dir, filename)
            print(f"\n=== 处理PDF文件: {filename} ===")
            pdf_chunks = collect_chunks_from_pdf(file_path)
            chunks.extend(pdf_chunks)
    
    print(f"\n总共处理了 {len(chunks)} 个PDF文件")
    return chunks

# 修改代码将PDF文件内容作为整体文本块
def collect_chunks_from_pdf(file_path: str):
    """处理单个PDF文件，将整个PDF内容作为一个文本块"""
    chunks: list[str] = []
    try:
        with open(file_path, 'rb') as file:
            # 使用PyPDF2库的PdfReader类读取PDF文件
            reader = PyPDF2.PdfReader(file)
            # 初始化一个空字符串，用于存储所有页面的文本内容
            text = ""
            # 提取文件名作为块的标题
            file_name = os.path.basename(file_path)
            # 添加标题格式，以便索引时能够提取
            text = f"# {file_name}\n\n"
            
            # 遍历PDF中的每一页
            for page in reader.pages:
                # 提取当前页的文本内容，并添加换行符以分隔页面
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            
            # 将整个PDF内容作为一个块添加到chunks列表
            if text.strip():
                chunks.append(text)
                print(f"已提取PDF: {file_name}，内容长度: {len(text)} 字符")
            else:
                print(f"警告: PDF文件 {file_name} 未提取到文本内容")
    except Exception as e:
        print(f"处理PDF文件 {file_path} 时出错: {str(e)}")
    
    return chunks

def collect_chunks(dir_or_file: str):
    if os.path.isdir(dir_or_file):
        return collect_chunks_from_dir(dir_or_file)
    return collect_chunks_from_file(dir_or_file)


def build_index_on_chunks(
    db: VecDB, key: str, chunks: list[str], batch_size: int = 100
):
    batch_size = 64
    model = get_embed_model()
    dim = model.get_sentence_embedding_dimension()
    assert isinstance(dim, int), "Cannot get embedding dimension"

    db.create_table_if_not_exists(key, dim)

    print(f"\n★ 开始构建索引：{key}")
    print(f"总块数：{len(chunks)} | 嵌入维度：{dim}D")
    print("首批次标题示例：", [chunk.split("\n")[0] for chunk in chunks[:3]])  # 显示前3个标题

    for i in tqdm(range(0, len(chunks), batch_size)):
        i_end = min(len(chunks), i + batch_size)
        content = chunks[i:i_end]
        title = [chunk.split("\n")[0].replace('# ', '') for chunk in content]
        vecs = model.encode(title, normalize_embeddings=True)
        db.batch_add(
            key,
            vecs.tolist(),
            [
                {"content": content, "title": title}
                for title, content in zip(title, content)
            ],
        )
        if i == 0:  # 仅首次迭代打印细节
            print("\n首批次数据样例：")
            print("标题向量：", vecs[0][:5], "...")  # 显示前5维
            print("元数据：", [c["content"][:30]+"..." for c in [{"content": content, "title": title} for title, content in zip(title, content)][:2]])


def build_index(db: VecDB, key: str, dir_or_file: str):
    chunks = collect_chunks(dir_or_file)
    build_index_on_chunks(db, key, chunks)
