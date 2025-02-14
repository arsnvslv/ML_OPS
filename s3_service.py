# s3_service.py
import boto3
import os

S3_BUCKET_NAME = "ml-bucket"  # Имя бакета Minio
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "admin123")

s3_client = boto3.client(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY,
    region_name="us-east-1",  # Миню не требует реальный регион, но пусть будет
)

def create_bucket():
    """Создаём бакет, если он ещё не существует."""
    buckets = [b["Name"] for b in s3_client.list_buckets()["Buckets"]]
    if S3_BUCKET_NAME not in buckets:
        s3_client.create_bucket(Bucket=S3_BUCKET_NAME)

def upload_to_s3(local_path: str, s3_key: str):
    """Загрузить локальный файл в S3 (Minio) по заданному ключу."""
    s3_client.upload_file(local_path, S3_BUCKET_NAME, s3_key)

def download_from_s3(s3_key: str, local_path: str):
    """Скачать файл из S3 (Minio) по ключу s3_key."""
    s3_client.download_file(S3_BUCKET_NAME, s3_key, local_path)

def get_list_from_bucket(prefix: str = ""):
    """Получить список объектов в бакете, фильтруя по префиксу."""
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=prefix)
    if "Contents" not in response:
        return []
    return [obj["Key"] for obj in response["Contents"]]
