import json
import logging
import os

import httpx
from fastapi import Form
from fastapi.responses import JSONResponse
from fastapi_poe.client import get_bot_response, get_final_response, QueryRequest,ToolDefinition
from fastapi_poe.types import ProtocolMessage
import uuid
import time

timeout = 120

logger = logging.getLogger("poe2openai")

client_dict = {}


async def get_responses(api_key, prompt=[], bot="gpt-4"):
    bot_name = get_bot(bot)
    # "system", "user", "bot"
    messages,tools = openai_message_to_poe_message(prompt)
    logger.info("=================", messages, "=================")

    additional_params = {"temperature": 0.7, "skip_system_prompt": False, "logit_bias": {}, "stop_sequences": []}
    query = QueryRequest(
        query=messages,
        user_id="",
        conversation_id="",
        message_id="",
        version="1.0",
        type="query",
        tools=tools if len(tools) > 0 else None,
        **additional_params
    )

    session = create_client()
    return await get_final_response(query, bot_name=bot_name, api_key=api_key, session=session)


# async def stream_get_responses(api_key, prompt, bot):
#     bot_name = get_bot(bot)
#     messages = openai_message_to_poe_message(prompt)

#     session = create_client()
#     async for partial in get_bot_response(messages=messages, bot_name=bot_name, api_key=api_key,
#                                           skip_system_prompt=True, session=session):
#         yield partial.text
async def stream_get_responses(api_key, prompt, bot):
    bot_name = get_bot(bot)
    messages,tools = openai_message_to_poe_message(prompt)
    if len(tools) == 0:
        tools = None
    session = create_client()
    async for partial in get_bot_response(messages=messages, bot_name=bot_name, api_key=api_key,tools=tools,
                                          skip_system_prompt=True, session=session):
        yield partial

def format_tool_calls(func_calls):
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "poe-bot",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{i}_{uuid.uuid4()}",
                            "type": "function",
                            "function": {
                                "name": call.function_name,
                                "arguments": call.arguments
                            }
                        } for i, call in enumerate(func_calls)
                    ]
                },
                "finish_reason": "tool_calls"
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }


def add_token(token: str):
    if token not in client_dict:
        try:
            client_dict[token] = token
            return "ok"
        except Exception as exception:
            logger.info("Failed to connect to poe due to " + str(exception))
            return "failed: " + str(exception)
    else:
        return "exist"


def get_bot(model):
    model_mapping = json.loads(os.environ.get("MODEL_MAPPING", "{}"))
    return model_mapping.get(model, "GPT-4o")


def openai_message_to_poe_message(openaiData):
    messages = openaiData['messages']
    new_messages = []
    for message in messages:
        role = message["role"]
        if role == "assistant":
            role = "bot"
        
        if role == "user" and isinstance(message["content"], list):
            # 处理多个用户消息
            for item in message["content"]:
                if item.get("type") == "text":
                    new_messages.append(ProtocolMessage(role=role, content=item["text"]))
        else:
            new_messages.append(ProtocolMessage(role=role, content=message["content"]))
    tools = []

    if 'tools' in openaiData:
        for tool in openaiData['tools']:
            tools.append(ToolDefinition.from_dict(tool['function']))
    return new_messages,tools

# def openai_message_to_poe_message(messages=[]):
#     new_messages = []
#     for message in messages:
#         role = message["role"]
#         if role == "assistant": role = "bot"
#         new_messages.append(ProtocolMessage(role=role, content=message["content"]))

#     return new_messages


def create_client():
    proxy_config = {
        "proxy_type": os.environ.get("PROXY_TYPE"),
        "proxy_host": os.environ.get("PROXY_HOST"),
        "proxy_port": os.environ.get("PROXY_PORT"),
        "proxy_username": os.environ.get("PROXY_USERNAME"),
        "proxy_password": os.environ.get("PROXY_PASSWORD"),
    }

    proxy = create_proxy(proxy_config)
    client = httpx.AsyncClient(timeout=600, proxies=proxy)
    return client


def create_proxy(proxy_config):
    proxy_type = proxy_config["proxy_type"]
    proxy_url = create_proxy_url(proxy_config)

    if proxy_type in ["http", "socks"] and proxy_url:
        return {
            "http://": proxy_url,
            "https://": proxy_url,
        }
    else:
        return None


def create_proxy_url(proxy_config):
    proxy_type = proxy_config["proxy_type"]
    proxy_host = proxy_config["proxy_host"]
    proxy_port = proxy_config["proxy_port"]
    # proxy_username = proxy_config["proxy_username"]
    # proxy_password = proxy_config["proxy_password"]

    if not proxy_host or not proxy_port:
        return None

    if proxy_type == "http":
        return f"http://{proxy_host}:{proxy_port}"
    elif proxy_type == "socks":
        return f"socks5://{proxy_host}:{proxy_port}"
    else:
        return None
