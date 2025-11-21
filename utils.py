from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.utilities import WikipediaAPIWrapper
import os
from typing import Optional


def generate_script(
        subject: str,
        video_length: float = 1.0,
        creativity: float = 0.7,
        api_key: Optional[str] = None
) -> tuple[str, str, str]:
    """
    基于 DeepSeek API 和维基百科搜索，生成年轻化风格的短视频脚本
    :param subject: 视频主题（如 "sora模型"、"可再生能源"）
    :param video_length: 视频时长（分钟），默认1分钟
    :param creativity: 生成创造力（0-1.5，越高越灵活，默认0.7）
    :param api_key: DeepSeek API 密钥（优先传入，无则读取环境变量）
    :return: 维基百科搜索结果、视频标题、结构化视频脚本
    """
    # 验证 API 密钥（避免空值调用）
    api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量，或直接传入 api_key 参数")

    # 标题生成模板（强化吸引力，适配短视频传播）
    title_template = ChatPromptTemplate.from_messages(
        [
            ("human", "请为'{subject}'主题的短视频创作1个吸引人的标题，严格遵循：\n"
                      "1. 用年轻人熟悉的网络热词/疑问句/反差感（例：'炸了！Sora居然能做电影级视频？'）\n"
                      "2. 20字以内，无专业术语，一眼抓注意力\n"
                      "3. 适配抖音/小红书等短视频平台传播逻辑")
        ]
    )

    # 脚本生成模板（优化结构清晰度，强化口语化风格）
    script_template = ChatPromptTemplate.from_messages(
        [
            ("human",
             """你是抖音/小红书风格的年轻向短视频博主，说话接地气、有网感，避免说教式表达。
             请根据以下信息生成结构化视频脚本，严格遵守要求：

             核心约束：
             - 视频标题：{title}
             - 目标时长：{duration}分钟（1分钟≈200字，总字数控制在 {word_count} 字左右）
             - 参考资料：维基百科搜索结果（仅提取相关干货，无关内容直接忽略）

             脚本结构要求（必须明确分隔）：
             1. 【开头】（30字内）：用反转/疑问/热点引入，瞬间抓住注意力（例："你敢信？AI视频已经卷到这种程度了！"）；
             2. 【中间】（核心干货）：提炼维基百科关键信息（技术原理/核心功能/应用场景），用大白话解释，无专业术语；
             3. 【结尾】（30字内）：留悬念/引导互动（例："下期实测Sora生成视频，评论区蹲链接的优先安排！"）；

             风格要求：
             - 全程口语化，像和朋友聊天，适当用表情符号（如🤯、🔥、🚀）增强感染力；
             - 避免长句，每句不超过15字，符合短视频快节奏表达；
             - 网络热词自然融入（如"卷疯了"、"YYDS"、"破防了"），不堆砌。

             参考资料：
             ```{wikipedia_search}```""")
        ]
    )

    # 初始化 DeepSeek 模型（配置超时重试，提升稳定性）
    model = ChatOpenAI(
        openai_api_key=api_key,
        temperature=creativity,
        base_url="https://api.deepseek.com/v1",  # 补充 v1 路径，避免 API 调用失败
        model="deepseek-chat",
        timeout=30,  # 延长超时时间，适配网络波动
        max_retries=2  # 失败自动重试2次
    )

    # 生成视频标题（链式调用，确保与主题强相关）
    title_chain = title_template | model
    title = title_chain.invoke({"subject": subject}).content.strip()

    # 维基百科搜索（优化异常处理，提升用户体验）
    search_result = ""
    try:
        wikipedia_api = WikipediaAPIWrapper(
            lang="zh",  # 中文词条优先
            timeout=15,  # 延长超时，适配维基百科国际访问
            extract_format="plaintext",
            sentences=10,  # 限制结果长度，避免冗余
            wiki_base_url="https://zh.wikipedia.org/w/api.php"  # 明确中文维基 API 地址，提升搜索成功率
        )
        search_result = wikipedia_api.run(subject)
        # 处理无结果场景
        if not search_result.strip():
            search_result = "维基百科未找到相关详细信息，以下基于公开常识生成内容"
    except ConnectionError:
        search_result = "维基百科网络连接失败，以下基于公开常识生成内容"
    except TimeoutError:
        search_result = "维基百科搜索超时，以下基于公开常识生成内容"
    except Exception as e:
        # 脱敏异常信息，避免泄露敏感内容
        search_result = f"维基百科搜索异常：{str(e)[:50]}...，以下基于公开常识生成内容"

    # 生成结构化视频脚本
    word_count = int(video_length * 200)
    script_chain = script_template | model
    script = script_chain.invoke({
        "title": title,
        "duration": video_length,
        "word_count": word_count,
        "wikipedia_search": search_result
    }).content.strip()

    return search_result, title, script


# # 本地测试入口（便于本地验证功能，简历中可体现测试意识）
# if __name__ == "__main__":
#     # 建议通过环境变量加载 API 密钥，避免硬编码泄露
#     api_key = os.getenv("DEEPSEEK_API_KEY") or "your-test-api-key"
#
#     try:
#         search_res, video_title, video_script = generate_script(
#             subject="sora模型",
#             video_length=1.0,
#             creativity=0.8,
#             api_key=api_key
#         )
#
#         # 格式化输出，提升可读性
#         print("=" * 60)
#         print(f"📚 维基百科搜索结果：\n{search_res}\n")
#         print(f"🔥 视频标题：{video_title}\n")
#         print(f"📝 视频脚本：\n{video_script}")
#         print("=" * 60)
#     except ValueError as ve:
#         print(f"❌ 配置错误：{ve}")
#     except Exception as e:
#         print(f"❌ 运行错误：{str(e)[:100]}...")