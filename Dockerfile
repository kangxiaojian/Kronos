# ==============================================================================
# Stage 1: Builder - 在 GPU 兼容环境中安装依赖
# ==============================================================================
# [修改] 更换基础镜像为 NVIDIA 官方的 CUDA 镜像
# - cuda:11.8.0: 指定了 CUDA 版本
# - cudnn8: 包含了 cuDNN 8 库
# - devel: 开发版镜像，包含了完整的编译工具链和 Python，适合作为构建环境
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 as builder

# 安装 Python 和 pip
# NVIDIA 基础镜像不一定自带特定版本的 Python，我们手动安装
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# [修改] 安装 GPU 版本的 PyTorch 和其他核心库
# 我们不再依赖 requirements.txt 里的 torch，而是直接指定安装 CUDA 11.8 对应的版本
# 这确保了框架和底层 CUDA 库的兼容性
RUN pip install --no-cache-dir \
    torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118 \
    gradio pandas transformers accelerate scikit-learn plotly

# 复制并安装 requirements.txt 中除了 torch 之外的其他依赖（如果有的话）
# 建议将 torch 相关库从 requirements.txt 中移除，避免冲突
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


# ==============================================================================
# Stage 2: Final Image - 创建精简的 GPU 运行环境
# ==============================================================================
# [修改] 最终镜像也必须是 CUDA 的运行时镜像
# runtime 镜像比 devel 镜像小，因为它不包含编译工具，更适合生产部署
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 安装 Python 运行时
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 从 builder 阶段复制已安装的 Python 包
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# 复制项目代码
COPY . .

# 设置环境变量
ENV HF_HOME=/app/.cache/huggingface
ENV HF_HUB_CACHE=$HF_HOME/hub

# 预下载模型（对于 GPU 环境同样重要）
RUN python3 -c "from model import Kronos, KronosTokenizer; KronosTokenizer.from_pretrained('NeoQuasar/Kronos-Tokenizer-base'); Kronos.from_pretrained('NeoQuasar/Kronos-small')"

# 暴露函数计算要求的 9000 端口
EXPOSE 9000

# [修改] 使用 python3 启动应用
CMD ["python3", "web_demo/app.py"]