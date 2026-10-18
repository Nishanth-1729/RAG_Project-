# API + UI image (CPU or GPU hosts with the NVIDIA container toolkit).
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
WORKDIR /app
COPY requirements.txt pyproject.toml ./
RUN grep -v '^torch' requirements.txt > /tmp/req.txt && pip install --no-cache-dir -r /tmp/req.txt
COPY src ./src
COPY frontend ./frontend
COPY Main.py ./
RUN pip install --no-cache-dir -e . --no-deps
ENV PYTHONUNBUFFERED=1
EXPOSE 8100 8501
CMD ["python", "Main.py", "serve", "--host", "0.0.0.0", "--port", "8100"]
