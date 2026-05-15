import json
import os
import subprocess
os.environ['MINERU_MODEL_SOURCE'] = 'modelscope' 
import torch
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from molscribe import MolScribe
from rxnscribe import RxnScribe


# --- 新增：MinerU 导入（带容错处理）---
try:
    from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
    from magic_pdf.data.dataset import PymuDocDataset
    from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
    from magic_pdf.config.enums import SupportedPdfParseMethod
    MINERU_AVAILABLE = True
    print("MinerU 加载成功")
except ImportError as e:
    MINERU_AVAILABLE = False
    print(f"MinerU 未安装，PDF 解析功能将不可用。错误: {e}")

# --- 初始化 FastAPI 应用 ---
app = FastAPI(title="OpenChemIE 综合识别 API (MolScribe + RxnScribe + MinerU)")


# 允许跨域（如果你之后开发前端，这一步是必须的）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 路径与环境配置 ---
BASE_DIR = "/public/home/whtian/ChemImageAnalyser"
MOL_CKPT = os.path.join(BASE_DIR, "ckpts", "swin_base_char_aux_1m680k.pth")
RXN_CKPT = os.path.join(BASE_DIR, "ckpts", "pix2seq_reaction_full.ckpt")
UPLOAD_DIR = os.path.join(BASE_DIR, "api/temp_uploads")
PDF_TEMP_DIR = os.path.join(BASE_DIR, "api/temp_pdf")  # 新增：PDF 临时目录
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "api/static")), name="static")

# 创建必要的目录
for dir_path in [UPLOAD_DIR, PDF_TEMP_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# --- 模型全局加载 ---
# 定义为全局变量，方便各路由调用
mol_model = None
rxn_model = None

@app.on_event("startup")
async def load_models():
    global mol_model, rxn_model
    print(f"正在加载模型至设备: {DEVICE} ...")
    try:
        # 1. 加载 MolScribe
        mol_model = MolScribe(MOL_CKPT, device=DEVICE)
        print("MolScribe 加载成功！aaaaaaa")
        print(RXN_CKPT)
        print(MOL_CKPT)
        
        # 2. 加载 RxnScribe (复用已加载的 mol_model)
        rxn_model = RxnScribe(RXN_CKPT, device=DEVICE)
        print("RxnScribe 加载成功！")
    except Exception as e:
        print(f"模型初始化失败，请检查路径。错误信息: {e}")

# --- API 路由定义 ---

@app.get("/")
async def root():
    return {
        "status": "online",
        "models_loaded": {
            "molscribe": mol_model is not None,
            "rxnscribe": rxn_model is not None
        },
        "pdf_parser_available": MINERU_AVAILABLE
    }

# 1. 分子图片识别 (MolScribe)
@app.post("/predict/molecule")
async def predict_molecule(file: UploadFile = File(...)):
    if mol_model is None:
        raise HTTPException(status_code=500, detail="MolScribe 模型未就绪")
    
    file_path = os.path.join(UPLOAD_DIR, f"mol_{file.filename}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        output = mol_model.predict_image_file(file_path, return_atoms_bonds=True, return_confidence=True)
        os.remove(file_path)
        return {
            "status": "success",
            "type": "molecule",
            "data": output
        }
    except Exception as e:
        if os.path.exists(file_path): os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

# 2. 化学反应识别 (RxnScribe)
@app.post("/predict/reaction")
async def predict_reaction(file: UploadFile = File(...)):
    if rxn_model is None:
        raise HTTPException(status_code=500, detail="RxnScribe 模型未就绪")
    
    file_path = os.path.join(UPLOAD_DIR, f"rxn_{file.filename}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # RxnScribe 会自动调用内部的 MolScribe 进行子结构解析
        # 开启 molscribe=True 和 ocr=True 以获取详细结构
        results = rxn_model.predict_image_file(file_path, molscribe=True, ocr=True)
        os.remove(file_path)
        return {
            "status": "success",
            "type": "reaction",
            "data": results
        }
    except Exception as e:
        if os.path.exists(file_path): os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

# --- 新增：MinerU PDF 解析功能 ---

@app.post("/parse/pdf")
async def parse_pdf(file: UploadFile = File(...)):
    """
    使用 MinerU CLI 解析 PDF，返回原始 PDF 和 bbox 数据
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="请上传 PDF 文件")
    
    pdf_name = os.path.splitext(file.filename)[0]
    temp_dir = os.path.join(PDF_TEMP_DIR, pdf_name)
    output_dir = os.path.join(temp_dir, "output")
    
    # 创建目录
    os.makedirs(output_dir, exist_ok=True)
    
    pdf_path = os.path.join(temp_dir, file.filename)
    
    try:
        # 1. 保存上传的 PDF
        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"开始解析 PDF: {pdf_path}")
        print(f"输出目录: {output_dir}")
        
        # 2. 调用 MinerU CLI
        cmd = [
            "mineru",
            "-p", pdf_path,
            "-o", output_dir,
            "-b", "pipeline"
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3000,  # 5分钟超时
            env=os.environ.copy()
        )
        
        if result.returncode != 0:
            error_msg = f"MinerU 解析失败:\n{result.stderr}"
            print(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
        
        print(f"MinerU 执行成功")
        
        # 3. MinerU 输出目录结构: {output_dir}/{pdf_name}/hybrid_auto/
        auto_dir = os.path.join(output_dir, pdf_name, "auto")
        
        if not os.path.exists(auto_dir):
            raise HTTPException(
                status_code=500, 
                detail=f"MinerU 输出目录不存在: {auto_dir}"
            )
        
        # 4. 读取 middle.json（包含 bbox 坐标数据）
        middle_json_path = os.path.join(auto_dir, f"{pdf_name}_middle.json")
        middle_data = {}
        
        if os.path.exists(middle_json_path):
            with open(middle_json_path, "r", encoding="utf-8") as f:
                middle_data = json.load(f)
                print(f"成功读取 middle.json，包含 {len(middle_data.get('pdf_info', []))} 页数据")
        else:
            print(f"警告: 未找到 middle.json: {middle_json_path}")
        
        # 5. 复制原始 PDF 到静态目录供前端访问
        original_pdf = os.path.join(auto_dir, f"{pdf_name}_origin.pdf")
        pdf_url = None
        
        if os.path.exists(original_pdf):
            # 静态文件目录
            static_pdf_dir = os.path.join(BASE_DIR, "api/static/pdfs", pdf_name)
            os.makedirs(static_pdf_dir, exist_ok=True)
            
            # 复制 origin.pdf
            static_pdf_path = os.path.join(static_pdf_dir, f"{pdf_name}_origin.pdf")
            shutil.copy(original_pdf, static_pdf_path)
            pdf_url = f"/static/pdfs/{pdf_name}/{pdf_name}_origin.pdf"
            print(f"原始 PDF 已复制到: {static_pdf_path}")
            
            # 同时复制 layout PDF（带标注的版本）供下载
            layout_pdf = os.path.join(auto_dir, f"{pdf_name}_layout.pdf")
            if os.path.exists(layout_pdf):
                static_layout_path = os.path.join(static_pdf_dir, f"{pdf_name}_layout.pdf")
                shutil.copy(layout_pdf, static_layout_path)
        else:
            # 如果没有 origin.pdf，尝试使用上传的原始文件
            static_pdf_dir = os.path.join(BASE_DIR, "api/static/pdfs", pdf_name)
            os.makedirs(static_pdf_dir, exist_ok=True)
            static_pdf_path = os.path.join(static_pdf_dir, f"{pdf_name}_origin.pdf")
            shutil.copy(pdf_path, static_pdf_path)
            pdf_url = f"/static/pdfs/{pdf_name}/{pdf_name}_origin.pdf"
        
        # 6. 复制提取的图片到静态目录
        images_dir = os.path.join(auto_dir, "images")
        static_images_dir = os.path.join(BASE_DIR, "api/static/pdfs", pdf_name)
        os.makedirs(static_images_dir, exist_ok=True)
        
        image_files = {}
        if os.path.exists(images_dir):
            for img in os.listdir(images_dir):
                src_path = os.path.join(images_dir, img)
                dst_path = os.path.join(static_images_dir, img)
                if os.path.isfile(src_path):
                    shutil.copy(src_path, dst_path)
                    image_files[img] = f"/static/pdfs/{pdf_name}/{img}"
            print(f"已复制 {len(image_files)} 张图片到静态目录")
        
        # 7. 可选：清理临时文件（保留静态目录供前端访问）
        # shutil.rmtree(temp_dir, ignore_errors=True)  # 如需清理取消注释
        
        return {
            "status": "success",
            "filename": file.filename,
            "pdf_url": pdf_url,  # 原始 PDF 访问路径
            "page_data": middle_data,  # 包含 bbox 的 JSON 数据
            "images": image_files,  # 图片路径映射
            "images_base_url": f"/static/pdfs/{pdf_name}/",  # 图片基础 URL
            "output_dir": auto_dir,
            "page_count": len(middle_data.get("pdf_info", []))
        }
        
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="PDF 解析超时（超过5分钟）")
    except Exception as e:
        import traceback
        error_detail = f"PDF 解析失败: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/parse/pdf/extract")
async def extract_pdf_images(file: UploadFile = File(...)):
    """
    解析 PDF 并提取其中的图片（特别是化学反应图）
    返回图片列表，可进一步传给 /predict/reaction 分析
    """
    if not MINERU_AVAILABLE:
        raise HTTPException(status_code=503, detail="MinerU 未安装")
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="请上传 PDF 文件")
    
    pdf_name = os.path.splitext(file.filename)[0]
    pdf_path = os.path.join(PDF_TEMP_DIR, f"{pdf_name}.pdf")
    output_dir = os.path.join(PDF_TEMP_DIR, pdf_name)
    image_dir = os.path.join(output_dir, "images")
    
    try:
        # 保存 PDF
        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 创建输出目录
        os.makedirs(image_dir, exist_ok=True)
        
        # 解析并提取内容
        reader = FileBasedDataReader("")
        pdf_bytes = reader.read(pdf_path)
        dataset = PymuDocDataset(pdf_bytes)
        infer_result = dataset.apply(doc_analyze, ocr=True)
        
        # 生成带图片路径的 Markdown（会自动保存图片到 image_dir）
        markdown_content = infer_result.get_markdown(image_dir)
        
        # 收集提取的图片文件
        extracted_images = []
        if os.path.exists(image_dir):
            for img_file in sorted(os.listdir(image_dir)):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(image_dir, img_file)
                    extracted_images.append({
                        "filename": img_file,
                        "local_path": img_path,
                        "relative_path": f"/temp/pdf/{pdf_name}/images/{img_file}"
                    })
        
        # 暂不删除 PDF 和图片，保留供后续可能的分析使用
        # 如需清理，可手动调用 DELETE 接口或设置定时任务
        
        return {
            "status": "success",
            "filename": file.filename,
            "output_directory": output_dir,
            "markdown": markdown_content,
            "images": extracted_images,
            "image_count": len(extracted_images)
        }
        
    except Exception as e:
        # 出错时清理
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        raise HTTPException(status_code=500, detail=f"PDF 处理失败: {str(e)}")

@app.get("/health")
async def health_check():
    """服务健康检查"""
    return {
        "status": "online",
        "models": {
            "molscribe_loaded": mol_model is not None,
            "rxnscribe_loaded": rxn_model is not None,
        },
        "pdf_parser": {
            "available": MINERU_AVAILABLE,
            "ready": MINERU_AVAILABLE  # MinerU 无需预加载，随用随调
        }
    }

# --- 启动服务 ---
if __name__ == "__main__":
    import uvicorn
    print(f"API 服务启动中...")
    print(f"MolScribe: {'就绪' if os.path.exists(MOL_CKPT) else '权重文件缺失'}")
    print(f"RxnScribe: {'就绪' if os.path.exists(RXN_CKPT) else '权重文件缺失'}")
    print(f"MinerU: {'就绪' if MINERU_AVAILABLE else '未安装'}")
    print("访问 http://localhost:9000/docs 查看接口文档")
    uvicorn.run(app, host="0.0.0.0", port=9070)


