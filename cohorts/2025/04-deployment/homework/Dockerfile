# 1) Use the pre-built model+vectorizer image as your base
FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

# 2) Create & set your working directory
WORKDIR /app

# 3) Copy in only your requirements first (to leverage build-cache)
COPY requirements.txt ./

# 4) Install all Python dependencies at once
RUN pip install --no-cache-dir -r requirements.txt

# 5) Copy your application code
COPY starter.py ./
COPY yellow_tripdata_2023-03.parquet ./

# 6) Define default command (adjust args as needed)
CMD ["python3", "starter.py", "2023", "3"]