import json
import logging
import os
from datetime import datetime

from dotenv import load_dotenv
from fastapi import APIRouter
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
import time

from fastapi.responses import JSONResponse, StreamingResponse

from api import poe_api
from fastapi_poe.client import BotMessageWithFunction
from util import utils

app = FastAPI()
logger = logging.getLogger(__name__)

router = APIRouter()
load_dotenv()


class Model(BaseModel):
    id: str
    object: str
    created: int
    owned_by: str

class ModelList(BaseModel):
    object: str
    data: List[Model]


@router.get("/v1/models")
async def get_models_from_env():
    logger.info('/v1/models is called')
    model_mapping = json.loads(os.environ.get("MODEL_MAPPING", "{}"))
    current_time = int(time.time())
    return {"data":[
        Model(
            id=name,
            object="model",
            created=current_time,
            owned_by='kbaicai'
        )
        for name, realname in model_mapping.items()
    ]}

@router.get("/")
async def root():
    return {"message": "Hello World"}




@router.post("/v1/chat/completions")
async def chat_proxy(request: Request):
    body = await request.json()
    logger.info(f'/v1/chat/completions is called with body{body}')

    model, messages, stream,tools = parse_request_body(body)
    if model is None:
        return JSONResponse(content={"error": "Invalid request body"}, status_code=400)

    token = await get_token_from_request(request)

    if stream:
        return StreamingResponse(process_openai_response_event_stream(model, body, token),
                                 media_type="text/event-stream")
    else:
        return await default_response(model, body, token)


def parse_request_body(body):
    try:
        model = body.get('model', 'gpt-3.5-turbo')
        messages = body.get('messages', [])
        stream = body.get('stream', False)
        tools = body.get('tools',None)

        return model, messages, stream,tools
    except json.JSONDecodeError as e:
        logger.debug(f"请求体解析错误: {e}")
        return None, None, None


async def get_token_from_request(request_data):
    logger.info("请求头: %s", request_data.headers)
    logger.info("开始获取token")
    token = request_data.headers.get('Authorization', '').replace('Bearer ', '')

    # 自定义token
    custom_token = os.environ.get('CUSTOM_TOKEN')
    # 内置token
    system_token = os.environ.get('SYSTEM_TOKEN')

    if token == custom_token:
        logger.info('使用的 token:{system_token}')
        return system_token

    return token


async def process_openai_response_event_stream(model, openaiRequestData, token):
    async for botMessage in poe_api.stream_get_responses(token, openaiRequestData, model):
        textcontent = botMessage.text
        result_line = f"data: {json.dumps(web_response_to_api_response_stream(botMessage, model))}\n\n"
        yield result_line
    # 通知结束
    yield f"data: {json.dumps(web_response_to_api_response_stream(None, model, True))}\n\n"
    # 通知结束
    yield "data: [DONE]\n\n"



#非函数式（流式）
#  {
#     "id": "chatcmpl-123",
#     "object": "chat.completion.chunk",
#     "created": 1694268190,
#     "model": "gpt-4o-mini",
#     "system_fingerprint": "fp_44709d6fcb",
#     "choices": [
#         {
#             "index": 0,
#             "delta": {
#                 "content": "Hello"
#             },
#             "logprobs": null,
#             "finish_reason": null
#         }
#     ]
# }


#函数式（流式）
    # data: {
    #     "id": "a34ca969-0024-4022-92e8-ddf7a1ce76a6",
    #     "object": "chat.completion.chunk",
    #     "created": 1727235856,
    #     "model": "deepseek-chat",
    #     "system_fingerprint": "fp_1c141eb703",
    #     "choices": [
    #         {
    #             "index": 0,
    #             "delta": {
    #                 "role": "assistant",
    #                 "content": ""
    #             },
    #             "logprobs": null,
    #             "finish_reason": null
    #         }
    #     ]
    # }

    # data: {
    #     "id": "a34ca969-0024-4022-92e8-ddf7a1ce76a6",
    #     "object": "chat.completion.chunk",
    #     "created": 1727235856,
    #     "model": "deepseek-chat",
    #     "system_fingerprint": "fp_1c141eb703",
    #     "choices": [
    #         {
    #             "index": 0,
    #             "delta": {
    #                 "tool_calls": [
    #                     {
    #                         "index": 0,
    #                         "id": "call_0_ef7cd47e-10dd-42f6-a8bd-fee404e5b51d",
    #                         "type": "function",
    #                         "function": {
    #                             "name": "get_current_weather",
    #                             "arguments": ""
    #                         }
    #                     }
    #                 ]
    #             },
    #             "logprobs": null,
    #             "finish_reason": null
    #         }
    #     ]
    # }

    # data: {
    #     "id": "a34ca969-0024-4022-92e8-ddf7a1ce76a6",
    #     "object": "chat.completion.chunk",
    #     "created": 1727235856,
    #     "model": "deepseek-chat",
    #     "system_fingerprint": "fp_1c141eb703",
    #     "choices": [
    #         {
    #             "index": 0,
    #             "delta": {
    #                 "tool_calls": [
    #                     {
    #                         "index": 0,
    #                         "function": {
    #                             "arguments": "{\""
    #                         }
    #                     }
    #                 ]
    #             },
    #             "logprobs": null,
    #             "finish_reason": null
    #         }
    #     ]
    # }

    # data: {
    #     "id": "a34ca969-0024-4022-92e8-ddf7a1ce76a6",
    #     "object": "chat.completion.chunk",
    #     "created": 1727235856,
    #     "model": "deepseek-chat",
    #     "system_fingerprint": "fp_1c141eb703",
    #     "choices": [
    #         {
    #             "index": 0,
    #             "delta": {
    #                 "tool_calls": [
    #                     {
    #                         "index": 0,
    #                         "function": {
    #                             "arguments": "location"
    #                         }
    #                     }
    #                 ]
    #             },
    #             "logprobs": null,
    #             "finish_reason": null
    #         }
    #     ]
    # }

    # data: {
    #     "id": "a34ca969-0024-4022-92e8-ddf7a1ce76a6",
    #     "object": "chat.completion.chunk",
    #     "created": 1727235856,
    #     "model": "deepseek-chat",
    #     "system_fingerprint": "fp_1c141eb703",
    #     "choices": [
    #         {
    #             "index": 0,
    #             "delta": {
    #                 "tool_calls": [
    #                     {
    #                         "index": 0,
    #                         "function": {
    #                             "arguments": "\":"
    #                         }
    #                     }
    #                 ]
    #             },
    #             "logprobs": null,
    #             "finish_reason": null
    #         }
    #     ]
    # }

    # data: {
    #     "id": "a34ca969-0024-4022-92e8-ddf7a1ce76a6",
    #     "object": "chat.completion.chunk",
    #     "created": 1727235856,
    #     "model": "deepseek-chat",
    #     "system_fingerprint": "fp_1c141eb703",
    #     "choices": [
    #         {
    #             "index": 0,
    #             "delta": {
    #                 "tool_calls": [
    #                     {
    #                         "index": 0,
    #                         "function": {
    #                             "arguments": " \""
    #                         }
    #                     }
    #                 ]
    #             },
    #             "logprobs": null,
    #             "finish_reason": null
    #         }
    #     ]
    # }

    # data: {
    #     "id": "a34ca969-0024-4022-92e8-ddf7a1ce76a6",
    #     "object": "chat.completion.chunk",
    #     "created": 1727235856,
    #     "model": "deepseek-chat",
    #     "system_fingerprint": "fp_1c141eb703",
    #     "choices": [
    #         {
    #             "index": 0,
    #             "delta": {
    #                 "tool_calls": [
    #                     {
    #                         "index": 0,
    #                         "function": {
    #                             "arguments": "Boston"
    #                         }
    #                     }
    #                 ]
    #             },
    #             "logprobs": null,
    #             "finish_reason": null
    #         }
    #     ]
    # }

    # data: {
    #     "id": "a34ca969-0024-4022-92e8-ddf7a1ce76a6",
    #     "object": "chat.completion.chunk",
    #     "created": 1727235856,
    #     "model": "deepseek-chat",
    #     "system_fingerprint": "fp_1c141eb703",
    #     "choices": [
    #         {
    #             "index": 0,
    #             "delta": {
    #                 "tool_calls": [
    #                     {
    #                         "index": 0,
    #                         "function": {
    #                             "arguments": "\"}"
    #                         }
    #                     }
    #                 ]
    #             },
    #             "logprobs": null,
    #             "finish_reason": null
    #         }
    #     ]
    # }

    # data: {
    #     "id": "a34ca969-0024-4022-92e8-ddf7a1ce76a6",
    #     "object": "chat.completion.chunk",
    #     "created": 1727235856,
    #     "model": "deepseek-chat",
    #     "system_fingerprint": "fp_1c141eb703",
    #     "choices": [
    #         {
    #             "index": 0,
    #             "delta": {
    #                 "content": ""
    #             },
    #             "logprobs": null,
    #             "finish_reason": "tool_calls"
    #         }
    #     ],
    #     "usage": {
    #         "prompt_tokens": 198,
    #         "completion_tokens": 23,
    #         "total_tokens": 221,
    #         "prompt_cache_hit_tokens": 0,
    #         "prompt_cache_miss_tokens": 198
    #     }
    # }

    # data: [DONE]

def poe_func_calls_to_openai(func_calls):
    function_calls = []
    for call in func_calls:
        function_call = {
            "name": call.function_name,
            "arguments": json.dumps(call.arguments)
        }
        function_calls.append({
            "index": 0,
            "id": f"call_{utils.get_8_random_str()}",
            "type": "function",
            "function": function_call
        })
    return function_calls

def web_response_to_api_response_stream(bot_message: BotMessageWithFunction, model: str, stop: bool = False):
    text_content = bot_message.text if bot_message else ''
    
    data = {
        "id": f"chatcmpl-{int(datetime.now().timestamp())}",
        "object": "chat.completion.chunk",
        "created": int(datetime.now().timestamp()),
        "model": model,
        "system_fingerprint": f"fp_{utils.get_8_random_str()}",
        "choices": [{
            "index": 0,
            "delta": {"content": f"{text_content}"},
            "finish_reason": "stop" if stop else None,
        }],
        "usage": {"prompt_tokens": 100, "completion_tokens": 100, "total_tokens": 100}
    }

    data["choices"][0]["delta"]["role"] = "assistant"
    if stop:
        return data

    if bot_message.function_calls:
        function_calls = poe_func_calls_to_openai(bot_message.function_calls)
        data["choices"][0]["delta"]["tool_calls"] = function_calls

    logger.info("openai 返回数据: %s", json.dumps(data, indent=2, ensure_ascii=False))

    return data



# def web_response_to_api_response_stream(text, model, stop=None):
#     data = {
#         "id": f"chatcmpl-{int(datetime.now().timestamp())}",
#         "object": "chat.completion.chunk",
#         "created": int(datetime.now().timestamp()),
#         "model": model,
#         "system_fingerprint": f"fp_{utils.get_8_random_str()}",
#         "choices": [{
#             "index": 0,
#             "delta": {"content": f"{text}"},
#             "finish_reason": "stop" if stop else None
#         }],
#         "usage": {"prompt_tokens": 100, "completion_tokens": 100, "total_tokens": 100}
#     }

#     logger.debug("openai 返回数据: %s", json.dumps(data, indent=2, ensure_ascii=False))

#     return data


async def default_response(model,openaiRequestData,token):
    textcontent = ''
    func_calls = []
    async for botMessage in poe_api.stream_get_responses(token, openaiRequestData, model):
        textcontent += botMessage.text
        if botMessage.function_calls:
            func_calls += botMessage.function_calls
    if len(func_calls) == 0:
        func_calls = None
    return web_response_to_api_response(model,textcontent,func_calls)

# async def default_response(model, messages, token):
    
#     result = await poe_api.get_responses(token, messages, model)

#     data = web_response_to_api_response(model, result)

#     return JSONResponse(content=data)


#非函数式
# {
#   "choices": [
#     {
#       "finish_reason": "stop",
#       "index": 0,
#       "message": {
#         "content": "The 2020 World Series was played in Texas at Globe Life Field in Arlington.",
#         "role": "assistant"
#       },
#       "logprobs": null
#     }
#   ],
#   "created": 1677664795,
#   "id": "chatcmpl-7QyqpwdfhqwajicIEznoc6Q47XAyW",
#   "model": "gpt-4o-mini",
#   "object": "chat.completion",
#   "usage": {
#     "completion_tokens": 17,
#     "prompt_tokens": 57,
#     "total_tokens": 74,
#     "completion_tokens_details": {
#       "reasoning_tokens": 0
#     }
#   }
# }

#函数式
# {
#   "id": "chatcmpl-abc123",
#   "object": "chat.completion",
#   "created": 1699896916,
#   "model": "gpt-4o-mini",
#   "choices": [
#     {
#       "index": 0,
#       "message": {
#         "role": "assistant",
#         "content": null,
#         "tool_calls": [
#             {
#                 "id": "call_abc123",
#                 "type": "function",
#                 "function": {
#                     "name": "get_current_weather",
#                     "arguments": "{\n\"location\": \"Boston, MA\"\n}"
#                 }
#             }
#         ]
#       },
#       "logprobs": null,
#       "finish_reason": "tool_calls"
#     }
#   ],
#   "usage": {
#     "prompt_tokens": 82,
#     "completion_tokens": 17,
#     "total_tokens": 99,
#     "completion_tokens_details": {
#       "reasoning_tokens": 0
#     }
#   }
# }

def web_response_to_api_response(model, text,func_calls):
    data = {
        "id": f"chatcmpl-{int(datetime.now().timestamp())}",
        "object": "chat.completion",
        "created": int(datetime.now().timestamp()),
        "model": model,
        "system_fingerprint": f"fp_{utils.get_8_random_str()}",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": f"{text}"},
            "logprobs": None,
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 100, "completion_tokens": 100, "total_tokens": 100}
    }
    if func_calls:
        function_calls = poe_func_calls_to_openai(func_calls)
        data['choices'][0]['message']['tool_calls'] = function_calls

        

    logger.info("openai 非流式返回数据: %s", json.dumps(data, indent=2, ensure_ascii=False))

    return data
