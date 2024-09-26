from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from route.route_chat import router as chat_router
import logging

logging.basicConfig(
    filename='./log/app.log',  # 确保这个路径与您的日志目录一致
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("poe2openai")


class CustomCORSMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        origin = request.headers.get('origin')
        if not origin:
            origin = "*"

        if request.method == "OPTIONS":
            response = Response()
            response.status_code = 204
            response.headers['Access-Control-Allow-Origin'] = origin
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Authorization, Content-Type, X-Requested-With'
            response.headers['Access-Control-Allow-Credentials'] = 'true'
            return response

        response = await call_next(request)
        # 检查请求来源，并动态设置响应头
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        response.headers['Access-Control-Allow-Methods'] = '*'
        response.headers['Access-Control-Allow-Headers'] = '*'

        return response

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code here
    logger.info("Starting up...")
    yield
    # Shutdown code he
    logger.info("Shutting down...")


app = FastAPI(lifespan=lifespan)

app.add_middleware(CustomCORSMiddleware)


app.include_router(chat_router)