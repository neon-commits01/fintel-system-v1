#Base Image - Starting filesystem
FROM python:3.13-slim
# Image working directory
WORKDIR /app
# copy the contents to the current working directory ( /app)
COPY requirements.txt .
# Install the dependencies listed in the reqs.txt
RUN pip install --no-cache-dir -r requirements.txt
# copy the local app folder to the image /app folder ->/app/app/main.py
COPY app/ app/
# copy models folder too -> /app/models/..
COPY models/ models/
# Inside the container, this app expects to listen on the port 8000
EXPOSE 8000

#CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}

