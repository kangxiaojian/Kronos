# ==============================================================================
# Stage 1: Builder - 在 GPU 兼容环境中安装依赖
# ==============================================================================
# [最终修正] 从您自己的阿里云私有仓库拉取基础镜像
FROM registry.cn-beijing.aliyuncs.com/memcloud/cuda:11.8.0-cudnn8-devel-ubuntu22.04 as builder

# --- 以下内容保持不变 ---
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118 \
    gradio pandas transformers accelerate scikit-learn plotly

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


# ==============================================================================
# Stage 2: Final Image - 创建精简的 GPU 运行环境
# ==============================================================================
# [最终修正] 最终镜像也从您自己的阿里云私有仓库拉取
FROM registry.cn-beijing.aliyuncs.com/memcloud/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# --- 以下内容保持不变 ---
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

COPY . .

ENV HF_HOME=/app/.cache/huggingface
ENV HF_HUB_CACHE=$HF_HOME/hub

RUN python3 -c "from model import Kronos, KronosTokenizer; KronosTokenizer.from_pretrained('NeoQuasar/Kronos-Tokenizer-base'); Kronos.from_pretrained('NeoQuasar/Kronos-small')"

EXPOSE 9000

CMD ["python3", "web_demo/app.py"]