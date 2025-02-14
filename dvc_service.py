import subprocess
import logging
import os
from fastapi import HTTPException

logger = logging.getLogger(__name__)

def run_dvc_command(command: list[str]) -> str:
    """
    Запускает DVC-команду (например, ["dvc", "add", "datasets/file.json"]).
    При ошибке выбрасывает HTTPException.
    Возвращает stdout.
    """
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        logger.info(f"DVC Command succeeded: {' '.join(command)}")
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logger.error(f"DVC Command failed: {' '.join(command)}\n{e.stderr}")
        raise HTTPException(status_code=500, detail=f"DVC command failed: {e.stderr}")

def dvc_add_push(filepath: str):
    """
    1) dvc add <filepath>
    2) dvc push
    """
    run_dvc_command(["dvc", "add", filepath])
    run_dvc_command(["dvc", "push"])
    logger.info(f"DVC add+push completed for {filepath}")

def dvc_pull():
    """
    dvc pull (вытягивает все файлы, у которых есть .dvc-файлы локально).
    """
    run_dvc_command(["dvc", "pull"])
    logger.info("DVC pull completed.")



