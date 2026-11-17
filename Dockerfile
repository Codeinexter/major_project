FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir pandas numpy scikit-learn pillow
COPY common.py client.py server.py /app/
ENTRYPOINT ["python"]
