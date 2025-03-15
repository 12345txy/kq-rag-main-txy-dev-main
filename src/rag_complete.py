import os
from typing import Iterable

from lab_1806_vec_db import VecDB
from openai import OpenAI

from src.embed_model import get_embed_model


def get_model_name():
    return os.getenv("MODEL_NAME") or "gpt-4o-mini"

def augment_prompt(query: str, db: VecDB, key: str, k=3, debug=False):
    embed_model = get_embed_model()
    input_embedding: list[float] = embed_model.encode([query])[0].tolist()
    results = db.search(key, input_embedding, k)
    if debug:
        print(f"Search with {k=}, {query=}; Got {len(results)} results")
        for idx, r in enumerate(results):
            title = r[0]["title"]
            print(f"{idx}: {title=}")

    source_knowledge = "\n".join([x[0]["content"] for x in results])
    print(f"Source Knowledge: {source_knowledge}")
    augmented_prompt = (
        f"使用提供的信息回答问题。\n\n信息:\n{source_knowledge}\n\n问题: \n{query}"
    )
    return augmented_prompt


def rag_complete(
    prompt: str,
    db: VecDB,
    key: str,
    model_name: str | None = None,
    debug=False,
) -> Iterable[str]:
    model_name = model_name or get_model_name()
    client = OpenAI(
    api_key = os.environ.get("ARK_API_KEY"),
    base_url = "https://ark.cn-beijing.volces.com/api/v3",
)
    print("----- streaming request -----")
    stream = client.chat.completions.create(
        model = "ep-20250216134646-bjpgx",  # your model endpoint ID
        messages = [
            {"role": "system", "content": "你是豆包，是由字节跳动开发的 AI 人工智能助手"},
            {"role": "user", "content": augment_prompt(prompt, db, key, debug=debug)},
        ],
        stream=True
    )

    for chunk in stream:
        choices = chunk.choices
        if len(choices) == 0:
            break
        content = choices[0].delta.content
        if content is not None:
            yield content
