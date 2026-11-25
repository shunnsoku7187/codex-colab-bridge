FROM xilinx/vitis-ai-pytorch-cpu@sha256:baf43fd9d6e1ea5a2c12b727ac3aafd15ead936172bcd636d7af98b0cca49298

RUN /opt/vitis_ai/conda/envs/vitis-ai-pytorch/bin/pip install --no-cache-dir --no-deps \
    timm==1.0.28 safetensors==0.4.5 huggingface_hub==0.24.7 \
    filelock==3.16.1 fsspec==2024.10.0 ninja==1.11.1.3 scipy==1.9.3
