import os
import httpx
import base64
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="化学文献解析网关")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置
BACKEND_URL = "http://192.168.10.86:9070"

# 千问 (Qwen) API 配置 - 你需要在阿里云百炼平台申请 API Key
# 申请地址: https://dashscope.aliyun.com/
QWEN_API_KEY = ""  # 替换为你的灵积 API Key
# sk-ea331d69caf444319747158304810bff
QWEN_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
QWEN_MODEL = "qwen-vl-plus"  # 多模态模型，支持图片理解

STATIC_DIR = "/public/home/whtian/ChemImageAnalyser/api/static"
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def read_root():
    return FileResponse('frontend/index.html')

@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{BACKEND_URL}/health")
            backend_status = "online" if resp.status_code == 200 else "error"
    except:
        backend_status = "unreachable"
    return {
        "status": "online",
        "backend": backend_status,
        "llm_service": "qwen-vl"
    }

@app.post("/parse/pdf")
async def parse_pdf_proxy(request: Request):
    """代理 PDF 解析请求到计算节点"""
    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ('host', 'content-length')}
    
    async with httpx.AsyncClient() as client:
        try:
            rp_resp = await client.post(
                f"{BACKEND_URL}/parse/pdf",
                headers=headers,
                content=body,
                timeout=3000.0
            )
            return Response(
                content=rp_resp.content,
                status_code=rp_resp.status_code,
                headers=dict(rp_resp.headers)
            )
        except Exception as e:
            raise HTTPException(500, f"后端请求失败: {str(e)}")

@app.post("/analyze/image")
async def analyze_image(request: Request):
    """
    分析图片内容：
    1. 调用 千问(Qwen-VL) 判断图片类型（分子/反应/其他）
    2. 根据类型调用对应后端服务
    """
    try:
        body = await request.json()
        image_url = body.get("image_url")
        image_path = body.get("image_path")
        
        if not image_url and not image_path:
            raise HTTPException(400, "需要提供 image_url 或 image_path")
        
        print(f"[Gateway] 收到图片分析请求: {image_path or image_url}")
        
        # 步骤 1: 使用 千问 判断图片类型
        image_type = await classify_with_qwen(image_url, image_path)
        print(f"[Gateway] 千问判断结果: {image_type}")
        
        # 步骤 2: 根据类型处理
        if image_type == "molecule":
            result = await call_molscribe(image_url, image_path)
            return {
                "type": "molecule",
                "analysis": "千问识别为分子结构图",
                "result": result
            }
        elif image_type == "reaction":
            result = await call_rxnscribe(image_url, image_path)
            return {
                "type": "reaction", 
                "analysis": "千问识别为化学反应式",
                "result": result
            }
        else:
            return {
                "type": "other",
                "analysis": "千问判断这不是化学结构图",
                "result": None,
                "suggestion": "请尝试点击图片中的化学结构区域"
            }
            
    except Exception as e:
        print(f"[Gateway] 分析失败: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"图片分析失败: {str(e)}")

async def classify_with_qwen(image_url: str, image_path: str) -> str:
    """
    调用 千问(Qwen-VL) API 判断图片类型
    """
    try:
        # 获取图片 base64
        image_base64 = None
        mime_type = "image/jpeg"
        
        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode('utf-8')
            ext = os.path.splitext(image_path)[1].lower().replace('.', '')
            if ext in ['jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp']:
                mime_type = f"image/{ext}" if ext != 'jpg' else "image/jpeg"
        else:
            # 从 URL 下载
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(image_url, timeout=30)
                image_base64 = base64.b64encode(resp.content).decode('utf-8')
        
        if not image_base64:
            return "molecule"  # 默认回退
        
        # 构建请求体 (OpenAI 兼容格式)
        payload = {
            "model": QWEN_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": "这是一张来自化学文献的图片。请判断这是：1.单个分子结构图(molecule)，2.化学反应式(reaction)，3.其他内容(other)。请只返回一个单词：molecule/reaction/other"
                        }
                    ]
                }
            ],
            "max_tokens": 50,
            "temperature": 0.1
        }
        
        # 调用 千问 API
        async with httpx.AsyncClient(timeout=30) as http_client:
            response = await http_client.post(
                QWEN_API_URL,
                headers={
                    "Authorization": f"Bearer {QWEN_API_KEY}",
                    "Content-Type": "application/json"
                },
                json=payload
            )
            
            print(f"[Qwen] API 状态码: {response.status_code}")
            
            if response.status_code == 401:
                error_text = await response.text()
                print(f"[Qwen] 认证失败: {error_text}")
                print("[Qwen] 请检查 QWEN_API_KEY 是否正确")
                return "molecule"
            
            if response.status_code != 200:
                error_text = await response.text()
                print(f"[Qwen] API 错误: {error_text}")
                return "molecule"  # 默认回退
            
            # 解析响应
            resp_data = response.json()
            print(f"[Qwen] 原始响应: {resp_data}")
            
            if "choices" not in resp_data or len(resp_data["choices"]) == 0:
                print("[Qwen] 响应中没有 choices")
                return "molecule"
            
            message_content = resp_data["choices"][0].get("message", {}).get("content", "")
            if not message_content:
                return "molecule"
            
            content_lower = message_content.lower().strip()
            print(f"[Qwen] 解析内容: {content_lower}")
            
            # 解析结果
            if "molecule" in content_lower or "分子" in content_lower:
                return "molecule"
            elif "reaction" in content_lower or "反应" in content_lower:
                return "reaction"
            elif "other" in content_lower or "其他" in content_lower:
                return "other"
            else:
                # 如果不确定，默认尝试分子（化学文献中分子图更常见）
                print(f"[Qwen] 无法明确识别，返回默认值: {content_lower}")
                return "molecule"
                
    except Exception as e:
        print(f"[Qwen] 调用失败: {e}")
        import traceback
        traceback.print_exc()
        return "molecule"  # 默认回退

async def call_molscribe(image_url: str, image_path: str):
    """调用计算节点的 MolScribe 服务"""
    try:
        async with httpx.AsyncClient(timeout=600) as client:
            if image_path and os.path.exists(image_path):
                with open(image_path, "rb") as f:
                    files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
                    resp = await client.post(
                        f"{BACKEND_URL}/predict/molecule",
                        files=files,
                        timeout=600
                    )
            else:
                img_resp = await client.get(image_url, timeout=300)
                files = {"file": ("image.jpg", img_resp.content, "image/jpeg")}
                resp = await client.post(
                    f"{BACKEND_URL}/predict/molecule",
                    files=files,
                    timeout=600
                )
            
            return resp.json()
    except Exception as e:
        raise Exception(f"MolScribe 调用失败: {str(e)}")

async def call_rxnscribe(image_url: str, image_path: str):
    """调用计算节点的 RxnScribe 服务"""
    try:
        async with httpx.AsyncClient(timeout=600) as client:
            if image_path and os.path.exists(image_path):
                with open(image_path, "rb") as f:
                    files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
                    resp = await client.post(
                        f"{BACKEND_URL}/predict/reaction",
                        files=files,
                        timeout=600
                    )
            else:
                img_resp = await client.get(image_url, timeout=300)
                files = {"file": ("image.jpg", img_resp.content, "image/jpeg")}
                resp = await client.post(
                    f"{BACKEND_URL}/predict/reaction",
                    files=files,
                    timeout=600
                )
            
            return resp.json()
    except Exception as e:
        raise Exception(f"RxnScribe 调用失败: {str(e)}")

# 静态文件和代理路由（保持不变）
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH", "TRACE"])
async def proxy(request: Request, path: str):
    """通用代理"""
    if path in ["", "health", "parse", "analyze"]:
        raise HTTPException(404, "Not found")
    
    url = httpx.URL(path=path, query=request.url.query.encode("utf-8"))
    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ('host', 'content-length')}
    
    async with httpx.AsyncClient() as client:
        try:
            rp_resp = await client.request(
                request.method,
                f"{BACKEND_URL}/{path}",
                headers=headers,
                content=body,
                timeout=3000.0,
                follow_redirects=True
            )
            return Response(
                content=rp_resp.content,
                status_code=rp_resp.status_code,
                headers=dict(rp_resp.headers)
            )
        except httpx.ConnectError as e:
            raise HTTPException(503, f"计算节点不可达: {str(e)}")
        except Exception as e:
            raise HTTPException(500, f"代理失败: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("=" * 50)
    print("化学文献解析网关")
    print(f"LLM 服务: 千问 (Qwen-VL)")
    print(f"API Key 状态: {'已配置' if QWEN_API_KEY else '未配置'}")
    print(f"后端地址: {BACKEND_URL}")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000)
