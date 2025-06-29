import datetime

import json
import os
import asyncio
import re

import sys
from typing import Dict, Any
import xml.etree.ElementTree as ET


from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.llm.zhipu import zhipu_embedding
from lightrag.utils import EmbeddingFunc
import numpy as np
from lightrag.kg.shared_storage import initialize_pipeline_status

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Current working directory:", os.getcwd())
WORKING_DIR = "./dickens"
os.environ["ZHIPUAI_API_KEY"] = "a68fdbc5e34345f3a5248336eee088e1.RVj0kFyUf88x4uDv"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def llm_model_func(
        prompt: str,
        system_prompt: str = None,
        history_messages: list = [],
        keyword_extraction: bool = False,
        **kwargs
) -> str:
    return await openai_complete_if_cache(
        "deepseek-chat",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key="sk-b77543c696c9446fab4b735de0442787",
        base_url="https://api.deepseek.com/v1",
        **kwargs,
    )


async def embedding_func(texts: list[str]) -> np.ndarray:
    return await zhipu_embedding(
        texts,
        "embedding-3",
    )


async def get_embedding_dim() -> int:
    test_text = ["This is a test sentence."]
    embedding = await embedding_func(test_text)
    return embedding.shape[1]


def extract_json_from_response(response: str) -> Dict[str, Any]:
    """从响应中提取有效的JSON内容"""
    # 尝试直接解析
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # 尝试去除Markdown代码块
    cleaned = re.sub(r'```(json)?|```', '', response).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 尝试提取第一个{...}之间的内容
    match = re.search(r'\{[\s\S]*\}', cleaned)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # 最终尝试：去除所有可能的非JSON内容
    lines = []
    in_json = False
    for line in cleaned.split('\n'):
        if line.strip().startswith('{') or in_json:
            in_json = True
            lines.append(line)
        if line.strip().endswith('}'):
            break
    final_attempt = '\n'.join(lines)
    try:
        return json.loads(final_attempt)
    except json.JSONDecodeError as e:
        raise ValueError(f"无法从响应中提取有效JSON: {e}\n原始响应:\n{response}")


def parse_graphml_nodes(graphml_file: str) -> Dict[str, Dict]:
    """解析GraphML文件中的节点信息"""
    tree = ET.parse(graphml_file)
    root = tree.getroot()

    # 定义XML命名空间
    ns = {'g': 'http://graphml.graphdrawing.org/xmlns'}

    nodes = {}
    for node in root.findall('.//g:node', ns):
        node_id = node.get('id')
        data_elements = node.findall('.//g:data', ns)

        node_data = {}
        for data in data_elements:
            key = data.get('key')
            text = data.text.strip() if data.text else ""
            node_data[key] = text

        nodes[node_id] = {
            "id": node_id,
            "label": node_data.get("d0", ""),  # 假设d0是标签键
            "description": node_data.get("d1", ""),  # 假设d1是描述键
            "type": node_data.get("d2", "concept"),  # 假设d2是类型键
            "source": node_data.get("d4","")
        }

    return nodes

async def initialize_rag() -> LightRAG:
    embedding_dimension = await get_embedding_dim()
    print(f"Detected embedding dimension: {embedding_dimension}")

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dimension,
            max_token_size=2048,
            func=embedding_func,
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag

async def main():
    try:
        rag = await initialize_rag()

        custom_prompt = """
        你是一个严格的 JSON 树结构生成器。请根据提供的节点信息构建一棵知识树，并将英文术语翻译成中文。输出格式必须如下所示：
        
        {history}
        
        {context_data}
        
        {{
          "id": "根节点名称（必须是专业知识点）",
          "entity_type": "类型",
          "level": "类型编号",
          "description": "中文描述",
          "style": {{"fill": "颜色代码"}},
          "source": "来源（默认为graphml，若无其他来源）",
          "children": [子节点]
        }}

        必须严格遵循以下规则：
        1. 基于提供的节点信息,每个节点可以重复利用,生成知识树；
        2. JSON 结构必须以 {{ 开头，以 }} 结尾，不能添加任何非 JSON 内容；
        3. 每个节点的 id 必须为**具体、专业的知识点术语**，例如 “深度学习”、“A*算法”，不能是如“print函数”、“numpy库”这类工具或库名称；
        4. 若节点来源为 unknown_source 或无来源，则 source 设为 “graphml”；
        5. description 和 entity_type 字段内容翻译为中文；
        6. 知识树根节点必须为 “Python”，其下可以包括所有与 Python 技术栈相关的专业知识点；
        7. 请尽可能保留所有具备专业知识意义的节点，除非节点类型为“组织”、“人物”或“地理”，这类节点应被过滤删除；
        8. 如果某些节点类型模糊，请优先保留，不要轻易删除；
        """

        context_data = []
        history_messages = []
        query_list = [
            "请根据我提供的节点信息生成符合格式的 JSON 树状结构，注意根节点必须是 Python。",
            "很好，现在请你在原来的基础上进行一次过滤，重新考虑每个节点对应的类型是否正确，最后把类型是‘组织’、‘人物’、‘地理’的节点都删除。",
            "接下来我们再做一次过滤，只保留那些具有专业知识意义的节点，比如‘A*算法’、‘深度学习’这类，而不是‘print函数’或‘numpy库’这种通用名称。注意，不能删太多节点，只要节点的 id 表达的是一个具体的知识概念，就应该保留下来。比如即使是小众算法，只要是专业术语就应该保留。每个节点的 id 必须是专业、独特的知识点，而不是某个工具、语言元素或通用库的名字。",
        ]

        graphml_file = os.path.join(WORKING_DIR, "graph_chunk_entity_relation.graphml")
        if not os.path.exists(graphml_file):
            raise FileNotFoundError(f"GraphML文件不存在: {graphml_file}")

        nodes = parse_graphml_nodes(graphml_file)
        context_data.append(nodes)

        for index, query in  enumerate(query_list):
            data = {"context_data":context_data,"history":history_messages}
            json_payload = json.dumps(data)
            try:
                response = await rag.aquery(
                    query=query,
                    param=QueryParam(mode="hybrid",max_token_for_text_unit = 6000,response_type= "Multiple Paragraphs"),
                    system_prompt= custom_prompt,
                )

                node_data = extract_json_from_response(response)
                context_data.append(node_data)
                history_messages.append({"role": "user", "content": query})
                history_messages.append({"role": "assistant", "content": node_data})

                if index == len(query_list)-1 :
                    nowTime = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
                    jsondir = "./json/" + nowTime + "_" + "tree_structure.json"
                    with open(jsondir, "w", encoding="utf-8") as f:
                        json.dump(node_data, f, indent=2, ensure_ascii=False)
                    await rag.aexport_data("./export_data/knowledge_graph.csv")
                    print("树状图JSON文件保存成功！")
            except ValueError as e:
                print(f"JSON处理失败: {e}")
                with open("failed_response.txt", "w", encoding="utf-8") as f:
                    f.write(response)

    except Exception as e:
        print(f"程序发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())