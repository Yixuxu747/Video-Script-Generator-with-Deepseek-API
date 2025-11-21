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
    api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量，或直接传入 api_key 参数")

    # 标题模板
    title_template = ChatPromptTemplate.from_messages(
        [
            ("human", "请为'{subject}'主题的短视频想1个吸引人的标题，要求：\n"
                      "1. 用年轻人熟悉的网络热词/疑问句/反差感（如：'炸了！Sora模型居然能做到这个？'）\n"
                      "2. 控制在20字以内，不堆砌专业术语")
        ]
    )

    # 脚本模板
    script_template = ChatPromptTemplate.from_messages(
        [
            ("human",
             """你是一位年轻向短视频博主，风格轻松有趣、说话接地气（多用网络热词，避免说教）。
             请根据以下信息生成视频脚本，严格遵循要求：

             核心信息：
             - 视频标题：{title}
             - 视频时长：{duration}分钟（1分钟≈200字，脚本总字数控制在 {word_count} 字左右）
             - 参考资料：维基百科搜索结果（仅用相关干货，无关内容直接忽略）

             脚本要求：
             1. 结构：必须用【开头】【中间】【结尾】三个部分明确分隔，每部分功能清晰；
             2. 开头（30字内）：用反转/疑问/热点引入（如："你敢信？AI视频现在已经卷到这种程度了！"）；
             3. 中间（核心干货）：提炼维基百科中的关键信息（如Sora的技术原理、核心功能、应用场景），用通俗语言解释，避免专业术语；
             4. 结尾（30字内）：留悬念/引导互动（如："下期实测Sora生成视频，评论区蹲链接的优先安排！"）；
             5. 风格：全程口语化，像和朋友聊天，可适当用表情符号（如🤯、🔥）增强感染力。

             参考资料：
             ```{wikipedia_search}```""")
        ]
    )

    # 初始化 DeepSeek 模型
    model = ChatOpenAI(
        openai_api_key=api_key,
        temperature=creativity,
        base_url="https://api.deepseek.com",
        model="deepseek-chat"   #换成需要的模型
    )

    # 生成标题
    title_chain = title_template | model
    title = title_chain.invoke({"subject": subject}).content.strip()

    # 维基百科搜索
    search_result = ""
    try:
        wikipedia_api = WikipediaAPIWrapper(
            lang="zh",
            timeout=10,
            extract_format="plaintext",
            sentences=10  # 限制结果长度，避免冗余
        )
        search_result = wikipedia_api.run(subject)
        if not search_result.strip():
            search_result = "维基百科未找到相关详细信息，以下基于公开常识生成内容"
    except Exception as e:
        search_result = f"维基百科搜索异常：{str(e)[:50]}...，以下基于公开常识生成内容"

    # 生成脚本
    word_count = int(video_length * 200)
    script_chain = script_template | model
    script = script_chain.invoke({
        "title": title,
        "duration": video_length,
        "word_count": word_count,
        "wikipedia_search": search_result
    }).content.strip()

    return search_result, title, script


# 测试运行（替换为你的 API 密钥）
# if __name__ == "__main__":
#     search_res, video_title, video_script = generate_script(
#         subject="sora模型",
#         video_length=1,
#         creativity=1.5,
#         api_key= os.getenv("DEEPSEEK_API_KEY") # 这里替换为实际的 DeepSeek API 密钥
#     )
#
#     # 格式化输出
#     print("=" * 50)
#     print(f"维基百科搜索结果：\n{search_res}\n")
#     print(f"视频标题：{video_title}\n")
#     print(f"视频脚本：\n{video_script}")
#     print("=" * 50)