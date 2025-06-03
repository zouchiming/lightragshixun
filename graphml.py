import json
import os
import asyncio
import re
import shutil
import sys
import logging
from typing import Dict, Any

from xlrd.xlsx import ET
import tempfile

from lightrag import LightRAG, QueryParam
from lightrag.llm import openai_complete_if_cache
from lightrag.llm.zhipu import zhipu_complete_if_cache, zhipu_embedding
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
import textract
from pdfminer.high_level import extract_text
from docx import Document

from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


print("Current working directory:", os.getcwd())
WORKING_DIR = "./dickens"
os.environ["ZHIPUAI_API_KEY"] = "a68fdbc5e34345f3a5248336eee088e1.RVj0kFyUf88x4uDv"

# 配置日志
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

# 腾讯云 COS 配置
secret_id = 'AKIDCOOCnCPPBk6GDyX3SecM7qR1VyGMPFuM'
secret_key = 'PcicKrDNL8j7HF1bx51kGYcYGn7E7mGZ'
region = 'ap-beijing'
bucket = 'pythonai-1354209443'
cos_folder = 'document/'

config = CosConfig(Region=region, SecretId=secret_id, SecretKey=secret_key)
client = CosS3Client(config)

def get_temp_dir():
    """获取有写入权限的临时目录"""
    try:
        # 尝试在程序目录创建
        app_temp = 'F:/gitcode/knowledgeGraph/LightRAG/temp_cos_pdf/'
        os.makedirs(app_temp, exist_ok=True)
        # 测试写入权限
        with open(os.path.join(app_temp, 'test.txt'), 'w') as f:
            f.write('test')
        os.remove(os.path.join(app_temp, 'test.txt'))
        return app_temp
    except (PermissionError, OSError):
        # 回退到系统临时目录
        sys_temp = os.path.join(tempfile.gettempdir(), 'cos_processing')
        os.makedirs(sys_temp, exist_ok=True)
        return sys_temp


def safe_textract_process(file_path):
    """更健壮的文本提取函数"""
    try:
        # 获取文件扩展名
        ext = os.path.splitext(file_path)[1].lower()

        # 明确指定文件类型
        if ext == '.pdf':
            result = textract.process(file_path, method='pdfminer', encoding='utf-8')
        elif ext == '.docx':
            result = textract.process(file_path, extension='docx', encoding='utf-8')
        elif ext == '.pptx':
            result = textract.process(file_path, extension='pptx', encoding='utf-8')
        else:
            result = textract.process(file_path, encoding='utf-8')

        return result.decode('utf-8') if result else ""

    except Exception as e:
        logging.warning(f"主提取失败: {file_path} - {type(e).__name__}: {str(e)}")

        # 备用方案：使用专用库
        try:
            if ext == '.pdf':

                return extract_text(file_path)
            elif ext == '.docx':

                doc = Document(file_path)
                return '\n'.join(p.text for p in doc.paragraphs)
            else:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
        except Exception as e:
            logging.error(f"备用方案失败: {file_path} - {type(e).__name__}: {str(e)}")
            return ""


async def load_and_process_cos_files():
    """从 COS 加载并处理文件（修复权限问题）"""
    # 获取安全的临时目录
    temp_dir = get_temp_dir()
    logging.info(f"使用临时目录: {temp_dir}")

    # 获取文件列表
    cos_files = []
    response = client.list_objects(Bucket=bucket, Prefix=cos_folder)
    if 'Contents' in response:
        cos_files = [item['Key'] for item in response['Contents'] if not item['Key'].endswith('/')]  # 过滤目录

    if not cos_files:
        logging.error("COS 中没有找到有效文件！")
        return None, None

    text_contents = []
    valid_files = []

    for cos_file in cos_files:
        temp_path = os.path.join(temp_dir, os.path.basename(cos_file))
        try:
            # 下载文件
            response = client.get_object(Bucket=bucket, Key=cos_file)
            with open(temp_path, 'wb') as f:
                shutil.copyfileobj(response['Body'].get_raw_stream(), f)

            # 处理文件
            text = safe_textract_process(temp_path)
            if text and text.strip():
                text_contents.append(text)
                valid_files.append(cos_file)
                logging.info(f"处理成功: {cos_file} (大小: {os.path.getsize(temp_path) / 1024:.2f}KB)")

        except Exception as e:
            logging.error(f"处理失败: {cos_file} - {type(e).__name__}: {str(e)[:200]}")
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except PermissionError:
                    logging.warning(f"无法删除临时文件: {temp_path}")

    return text_contents, valid_files

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

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=2048,
            max_token_size=2048,
            func=lambda texts: zhipu_embedding(texts),
        ),
        addon_params={
            "insert_batch_size": 4  # 每批处理4个文档
        }
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


async def main():
    try:
        rag =await initialize_rag()

        # 清除所有缓存
        await rag.aclear_cache()

        text_contents, cos_pdf_files = await load_and_process_cos_files()
        if text_contents:
            try:
                await rag.ainsert(text_contents, file_paths=cos_pdf_files)
                logging.info("成功插入 RAG 系统！")
            except Exception as e:
                logging.error(f"插入 RAG 失败: {e}")

        # custom_prompt = """
        #     你是一个严格的JSON生成器，并且将其中的英文翻译成中文，必须基于以下节点信息生成树状图结构：
        #    现有结点信息：
        #     {context_data}
        #
        #     {{
        #       "id": "根节点名称",
        #       "entity_type": "类型",
        #       "level":"类型编号"
        #       "description": "描述",
        #       "style": {{"fill": "颜色代码"}},
        #       “source": "来源",
        #       "children": [],
        #     }}
        #
        #     规则：
        #     1. 必须基于提供的节点信息
        #     2. 直接以{{开头，以}}结尾
        #     3. 不要有任何非JSON内容
        #     4. source若没有来源或来源为‘unknown_source’则默认为graphml,source对应现有节点的file_path
        #     5. children中的节点格式与根节点相同
        #     6. description中翻译为中文
        #     7. id必须简短且为中文,必须有特定含义,与父节点重合的内容可以省去
        #     8. entity_type不能有人物出现
        #     9. 根节点必须是Python
        #     """
        #
        # try:
        #
        #     response = await rag.aquery(
        #          "要求将现用转化为我要求的格式,注意是所有我提供的节点",
        #         param=QueryParam(mode="hybrid"),
        #         system_prompt= custom_prompt
        #     )
        #
        #     data = extract_json_from_response(response)
        #     node_data =  json.dumps(list(data.values()), indent=2, ensure_ascii=False)
        #
        #     response = await rag.aquery(
        #         "将已生成的图谱进行进一步过滤和筛选，要求每个节点的id都必须是具体的专业知识点而非普通名词或动作名称，删去类型为组织、人物、地理的节点",
        #         param=QueryParam(mode="hybrid"),
        #         system_prompt=custom_prompt
        #     )
        #
        #     data = extract_json_from_response(response)
        #     node_data =  json.dumps(list(data.values()), indent=2, ensure_ascii=False)
        #     response = await rag.aquery(
        #         "将已生成的图谱连接为树状",
        #         param=QueryParam(mode="hybrid"),
        #         system_prompt=custom_prompt
        #     )
        #
        #     print(response)
        #     data = extract_json_from_response(response)
        #     with open("tree_structure.json", "w", encoding="utf-8") as f:
        #         json.dump(data, f, indent=2, ensure_ascii=False)
        #     print("树状图JSON文件保存成功！")
        # except ValueError as e:
        #     print(f"JSON处理失败: {e}")
        #     with open("failed_response.txt", "w", encoding="utf-8") as f:
        #         f.write(response)

    except Exception as e:
        print(f"程序发生错误: {e}")
        import traceback
        traceback.print_exc()
if __name__ == "__main__":
    asyncio.run(main())