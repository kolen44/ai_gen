"""Жизненный цикл GPU-подов RunPod: создать → дождаться SSH → залить код → поднять воркер →
генерировать → удалить.

Остальной пайплайн ходит во внешние API, где нужен только ключ. Здесь карта арендуется и на неё
ставится наш воркер (runpod_worker/), поэтому и появляется весь цикл.

Подов может быть несколько: разные чекпойнты требуют разной VRAM, и переключать их на одной карте
значит платить за простой.

Реестр лежит в JSON рядом с прогонами, а не в памяти: админку перезапускают, а карта продолжает
тарифицироваться, и потерять её id — значит платить за под, о котором забыли.
"""

from __future__ import annotations

import json
import subprocess
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = ROOT / "runs" / "runpod_pods.json"
RUNPODCTL = ROOT / "runpodctl.exe"

GRAPHQL = "https://api.runpod.io/graphql"

# Порт воркера внутри пода. RunPod проксирует его наружу как <podId>-<port>.proxy.runpod.net —
# без отдельного публичного IP и без возни с firewall.
WORKER_PORT = 8000

# Образ с уже собранным torch под конкретную CUDA. Свой не собираем намеренно: сборка и заливка
# образа занимают больше времени, чем pip install на поде, а обновление кода потребовало бы
# пересборки вместо scp.
BASE_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"

# Профили карт. VRAM — единственное, что реально различает эти конфигурации: SDXL-ветка целиком
# помещается в 24 GB (замерено: 8.91 с на кадр 1536x1536 на RTX 4090), Chroma на архитектуре Flux
# требует заметно больше и на 24 GB идёт только с оффлоадом.
GPU_PROFILES = {
    "rtx4090": {"gpu_type": "NVIDIA GeForce RTX 4090", "vram_gb": 24, "disk_gb": 60,
                "label": "RTX 4090 24GB — SDXL-чекпойнты"},
    "a6000": {"gpu_type": "NVIDIA RTX A6000", "vram_gb": 48, "disk_gb": 80,
              "label": "RTX A6000 48GB — Chroma и всё остальное"},
    "l40s": {"gpu_type": "NVIDIA L40S", "vram_gb": 48, "disk_gb": 80,
             "label": "L40S 48GB — быстрее A6000"},
    # Диск 260 GB: сам Wan2.2-I2V-A14B в diffusers-формате весит 126 GB (два эксперта по 57 GB
    # плюс T5), fp16-вариантов у него нет. Плюс SDXL-чекпойнты и кеш HuggingFace — в 120 GB это
    # не помещается, проверено размерами через HF API.
    "a100": {"gpu_type": "NVIDIA A100 80GB PCIe", "vram_gb": 80, "disk_gb": 260,
             "label": "A100 80GB — видео A14B"},
}


# --- реестр ------------------------------------------------------------------

@dataclass
class PodRecord:
    """Что админка знает о поде между перезапусками.

    Только то, чего нельзя переспросить у RunPod: профиль, чекпойнт, состояние развёртывания.
    Статус, цена и наработка берутся из API — копия в файле разошлась бы с действительностью.
    """
    pod_id: str
    name: str
    profile: str
    photo_model: str
    identity_mode: str
    created_at: str
    deploy_state: str = "new"     # new | deploying | ready | failed
    deploy_log: str = ""
    ssh_host: Optional[str] = None
    ssh_port: Optional[int] = None


_registry_lock = threading.Lock()


def _load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        return {}
    try:
        raw = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {k: PodRecord(**v) for k, v in raw.items()}


def _save_registry(records: dict) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(
        json.dumps({k: asdict(v) for k, v in records.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def get_record(pod_id: str) -> Optional[PodRecord]:
    return _load_registry().get(pod_id)


def update_record(pod_id: str, **fields) -> None:
    with _registry_lock:
        records = _load_registry()
        if pod_id in records:
            for key, value in fields.items():
                setattr(records[pod_id], key, value)
            _save_registry(records)


def append_log(pod_id: str, line: str) -> None:
    with _registry_lock:
        records = _load_registry()
        if pod_id not in records:
            return
        # Держим хвост, а не весь лог: установка окружения печатает тысячи строк pip, целиком они
        # не нужны ни в браузере, ни в JSON-реестре.
        tail = (records[pod_id].deploy_log + line + "\n").splitlines()[-80:]
        records[pod_id].deploy_log = "\n".join(tail) + "\n"
        _save_registry(records)


# --- API RunPod --------------------------------------------------------------

def _api_key() -> str:
    from pipeline import prompt_config

    key = prompt_config._get(prompt_config.read_env_file(), "RUNPOD_API_KEY")
    if not key:
        raise RuntimeError("RUNPOD_API_KEY не задан — впишите его в разделе ключей")
    return key


def _graphql(query: str) -> dict:
    response = requests.post(f"{GRAPHQL}?api_key={_api_key()}", json={"query": query}, timeout=30)
    response.raise_for_status()
    data = response.json()
    if "errors" in data:
        raise RuntimeError(data["errors"][0].get("message", "ошибка RunPod API"))
    return data["data"]


def live_pods() -> list:
    """Поды, арендованные на аккаунте — источник правды по статусу и цене.

    Смотрим аккаунт целиком, а не только свой реестр: под, созданный из другого места или
    оставшийся от прошлой сессии, всё равно тарифицируется и должен быть виден.
    """
    data = _graphql(
        "{ myself { pods { id name desiredStatus costPerHr "
        "machine { gpuDisplayName } "
        "runtime { uptimeInSeconds } } } }"
    )
    return data["myself"]["pods"] or []


def gpu_catalog() -> list:
    """Доступность и цена профилей — чтобы не выбирать вслепую карту, которой сейчас нет."""
    data = _graphql(
        "{ gpuTypes { id displayName memoryInGb secureCloud "
        "lowestPrice(input:{gpuCount:1}) { uninterruptablePrice } } }"
    )
    by_id = {g["id"]: g for g in data["gpuTypes"]}
    out = []
    for key, profile in GPU_PROFILES.items():
        gpu = by_id.get(profile["gpu_type"])
        out.append({
            "key": key,
            "label": profile["label"],
            "gpu_type": profile["gpu_type"],
            "vram_gb": profile["vram_gb"],
            # Доступность — по наличию цены, а не по флагу secureCloud: тот говорит лишь «такая
            # карта в каталоге бывает». lowestPrice=None значит, что свободных нет, и аренда
            # падает с "no longer any instances available".
            "available": bool(gpu and gpu.get("secureCloud")
                              and (gpu.get("lowestPrice") or {}).get("uninterruptablePrice")),
            "price": (gpu or {}).get("lowestPrice", {}).get("uninterruptablePrice"),
        })
    return out


def create_pod(name: str, profile: str, photo_model: str, identity_mode: str) -> PodRecord:
    if profile not in GPU_PROFILES:
        raise RuntimeError(f"неизвестный профиль: {profile}")
    spec = GPU_PROFILES[profile]

    if not RUNPODCTL.exists():
        raise RuntimeError(f"не найден {RUNPODCTL.name} — скачайте runpodctl в корень проекта")

    # runpodctl, а не GraphQL-мутация: он сам генерирует ed25519-ключ и регистрирует его в
    # аккаунте, иначе SSH пришлось бы настраивать руками через веб-интерфейс.
    subprocess.run([str(RUNPODCTL), "config", "--apiKey", _api_key()],
                   capture_output=True, text=True, timeout=60)

    result = subprocess.run(
        [
            str(RUNPODCTL), "create", "pod",
            "--name", name,
            # Secure Cloud обязателен: на Community SSH идёт через прокси ssh.runpod.io, который
            # требует хеш из веб-интерфейса и не поддерживает scp — залить код нечем.
            "--secureCloud",
            "--gpuType", spec["gpu_type"], "--gpuCount", "1",
            "--imageName", BASE_IMAGE,
            "--containerDiskSize", str(spec["disk_gb"]),
            "--volumeSize", "20", "--volumePath", "/workspace",
            "--mem", "32", "--vcpu", "8",
            "--startSSH",
            # Максимум один http и один tcp: 22 для заливки кода, WORKER_PORT наружу под API.
            "--ports", f"22/tcp,{WORKER_PORT}/http",
        ],
        capture_output=True, text=True, timeout=180,
    )
    output = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        raise RuntimeError(f"создание пода не удалось: {output.strip()[:300]}")

    pod_id = _parse_pod_id(output)
    if not pod_id:
        raise RuntimeError(f"не удалось разобрать id пода из ответа: {output.strip()[:200]}")

    record = PodRecord(
        pod_id=pod_id, name=name, profile=profile, photo_model=photo_model,
        identity_mode=identity_mode, created_at=datetime.now(timezone.utc).isoformat(),
    )
    with _registry_lock:
        records = _load_registry()
        records[pod_id] = record
        _save_registry(records)
    return record


def _parse_pod_id(output: str) -> Optional[str]:
    """runpodctl печатает: pod "xzggzkstpxlfer" created for $0.440 / hr

    Разбираем по кавычкам, а не регуляркой по всей строке: в выводе рядом идут предупреждения
    про deprecated-команды, и общий поиск «слово из букв и цифр» цеплял бы их.
    """
    parts = output.split('"')
    for token in parts[1::2]:
        if len(token) >= 8 and token.isalnum():
            return token
    return None


def delete_pod(pod_id: str) -> None:
    """Удаление, а не остановка: остановленный под продолжает занимать деньги за том."""
    subprocess.run([str(RUNPODCTL), "remove", "pod", pod_id],
                   capture_output=True, text=True, timeout=120)
    with _registry_lock:
        records = _load_registry()
        records.pop(pod_id, None)
        _save_registry(records)


def ssh_info(pod_id: str) -> Optional[tuple]:
    """(host, port) прямого SSH или None, если контейнер ещё не поднялся.

    None здесь — обычное состояние первых минут, а не ошибка: RunPod сначала арендует машину,
    потом тянет образ, и только потом появляется контейнер с портом.
    """
    result = subprocess.run([str(RUNPODCTL), "ssh", "info", pod_id],
                            capture_output=True, text=True, timeout=60)
    try:
        data = json.loads(result.stdout)
    except ValueError:
        return None

    # Берём поля ip/port, а не разбираем ssh_command: там user@host стоит не вторым токеном, и
    # разбор по позиции давал host="-i". Плюс путь к ключу на Windows содержит пробелы.
    ip, port = data.get("ip"), data.get("port")
    if ip and port:
        return ip, int(port)

    # Запасной разбор — на случай, если в другой версии runpodctl отдельных полей не окажется.
    command = data.get("ssh_command")
    if not command:
        return None
    for token in command.split():
        if "@" in token:
            host = token.split("@")[-1]
            parts = command.split()
            port = int(parts[parts.index("-p") + 1]) if "-p" in parts else 22
            return host, port
    return None


def worker_url(pod_id: str) -> str:
    return f"https://{pod_id}-{WORKER_PORT}.proxy.runpod.net"


def worker_health(pod_id: str, timeout: float = 8.0) -> Optional[dict]:
    """None — воркер ещё не отвечает. Тоже нормальное состояние, а не ошибка: под может быть занят
    установкой окружения, а первая загрузка чекпойнта в память занимает минуты."""
    try:
        response = requests.get(f"{worker_url(pod_id)}/health", timeout=timeout)
        if response.status_code == 200:
            return response.json()
    except requests.RequestException:
        return None
    return None


# --- развёртывание -----------------------------------------------------------

# Заливаем только код: веса тянутся на поде прямо с HuggingFace, результаты уходят по HTTP.
# pipeline/ обязателен, хотя генерация в нём не живёт — оттуда SafetyPipeline читает пороги.
# Без него импорт молча падает в except, NSFW-порог откатывается к 0.5, и кадры бракует фильтр.
# Выглядит как «модель не рисует».
DEPLOY_DIRS = ["worker", "safety", "runpod_worker", "scripts", "pipeline"]

# Карточка едет вместе с кодом: без неё воркер не видит настроек админки и берёт запасные, а они
# намеренно строгие. Ключи в .env тоже есть, но под получает их отдельными переменными окружения —
# здесь они просто едут вместе с файлом.
DEPLOY_FILES = [".env"]

def _ssh_key() -> Path:
    """Приватный ключ для входа на под.

    Имя файла у runpodctl менялось между версиями, поэтому перебираем кандидатов: иначе после
    обновления развёртывание молча падает на «Permission denied».
    """
    candidates = [
        Path.home() / ".runpod" / "ssh" / "runpodctl-ssh-key",
        Path.home() / ".runpod" / "ssh" / "RunPod-Key-Go",
        Path.home() / ".ssh" / "runpod_synthetic_char",
        Path.home() / ".ssh" / "id_ed25519",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise RuntimeError(
        "не найден приватный SSH-ключ RunPod. Выполните: runpodctl.exe config --apiKey <ключ>"
    )

# Общие опции SSH. Проверку ключа хоста отключаем осознанно: под живёт часы и каждый раз имеет
# новый ключ, поэтому строгая проверка означала бы ручное подтверждение при каждом развёртывании,
# а known_hosts копился бы мусором. Пишем в null, чтобы файл не рос.
def _ssh_opts() -> list:
    return [
        "-i", str(_ssh_key()),
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        "-o", "ConnectTimeout=15",
        # Keepalive обязателен: генерация видео идёт минутами без байта в канале, и соединение
        # рвётся посреди запроса, хотя на поде ролик считается дальше. Пакет раз в 30 секунд,
        # разрыв — только после 240 пропущенных подряд (два часа).
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=240",
        # TCPKeepAlive выключаем намеренно: он полагается на TCP-уровень, который у NAT-прокси
        # между нами и подом может молча резать простаивающие соединения. ServerAlive работает на
        # уровне протокола SSH и такой проверке не подвержен.
        "-o", "TCPKeepAlive=no",
    ]


def _ssh(host: str, port: int, command: str, timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", *_ssh_opts(), "-p", str(port), f"root@{host}", command],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
    )


def wait_for_ssh(pod_id: str, attempts: int = 40, delay: int = 15) -> tuple:
    """Ждёт, пока RunPod поднимет контейнер и опубликует порт.

    Долгое ожидание здесь нормально: под сначала арендуется, потом тянется образ (~10 GB), и
    только потом появляется SSH. На практике 2-6 минут, изредка больше десяти.
    """
    import time

    for attempt in range(attempts):
        info = ssh_info(pod_id)
        if info:
            append_log(pod_id, f"SSH готов: {info[0]}:{info[1]}")
            return info
        if attempt % 4 == 0:
            append_log(pod_id, f"жду контейнер… ({attempt * delay // 60} мин)")
        time.sleep(delay)
    raise RuntimeError("под не поднял SSH за отведённое время")


def deploy_worker(pod_id: str) -> None:
    """Полное развёртывание: код → окружение → запущенный воркер.

    Выполняется в фоновом потоке, прогресс пишется в deploy_log записи пода, потому что целиком
    это занимает 10-20 минут и держать на нём HTTP-запрос от браузера нельзя.
    """
    import tarfile
    import tempfile

    record = get_record(pod_id)
    if record is None:
        raise RuntimeError("под не найден в реестре")

    from pipeline import prompt_config

    env = prompt_config.read_env_file()
    hf_token = prompt_config._get(env, "HF_TOKEN") or ""
    # Ключ доступа к воркеру. Порт публикуется наружу через прокси RunPod, то есть API виден
    # всему интернету — без ключа генерацию сможет запускать кто угодно, а GPU-часы оплачиваем мы.
    worker_key = prompt_config._get(env, "WORKER_API_KEY") or pod_id

    update_record(pod_id, deploy_state="deploying")
    try:
        host, port = wait_for_ssh(pod_id)
        update_record(pod_id, ssh_host=host, ssh_port=port)

        # --- освобождаем том до заливки ---
        # Порядок важен: /workspace — сетевой том с квотой, и если её занял кеш весов от прошлой
        # раскладки, то падает уже scp кода ("scp: close remote: Failure"), то есть развёртывание
        # умирает раньше, чем дошло бы до любой очистки. Найдено на реальном прогоне.
        _ssh(host, port, "rm -rf /workspace/hf /workspace/models /workspace/code.tar.gz "
                         "/workspace/app || true", timeout=300)

        # --- код ---
        append_log(pod_id, "упаковываю код…")
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "code.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                for name in DEPLOY_FILES:
                    source = ROOT / name
                    if source.exists():
                        tar.add(source, arcname=name)
                for name in DEPLOY_DIRS:
                    source = ROOT / name
                    if source.exists():
                        # __pycache__ исключаем: .pyc собраны для Windows-питона и на поде
                        # бесполезны, зато заметно раздувают архив.
                        tar.add(source, arcname=name,
                                filter=lambda t: None if "__pycache__" in t.name else t)
            append_log(pod_id, f"заливаю {archive.stat().st_size // 1024} КБ…")
            upload = subprocess.run(
                ["scp", *_ssh_opts(), "-P", str(port), str(archive), f"root@{host}:/root/code.tar.gz"],
                capture_output=True, text=True, timeout=600,
            )
            if upload.returncode != 0:
                raise RuntimeError(f"scp не прошёл: {upload.stderr.strip()[:200]}")

        _ssh(host, port, "mkdir -p /root/app && tar xzf /root/code.tar.gz -C /root/app")
        append_log(pod_id, "код на поде")

        # --- окружение ---
        # Порядок и версии не произвольны, см. deploy/runpod_setup.md п.3:
        # 1. torch — явной версией из cu124-индекса. В образе 2.4.1, а transformers требует 2.5+:
        #    иначе is_torch_available() возвращает False и модуль падает на NameError: nn.
        #    А "pip install -U torch" без индекса тянет сборку под CUDA 13, и CUDA отваливается.
        # 2. HF-стек — одной командой: отдельными вызовами pip не согласует версии.
        append_log(pod_id, "ставлю torch 2.6.0+cu124 (5-10 мин)…")
        torch_install = _ssh(host, port, (
            "pip install --no-cache-dir "
            "torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 "
            "--index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -3"
        ), timeout=2400)
        append_log(pod_id, torch_install.stdout.strip()[-300:] or "torch готов")

        append_log(pod_id, "ставлю HF-стек одной командой…")
        install = _ssh(host, port, (
            "pip install --no-cache-dir "
            "'fastapi>=0.115' 'uvicorn[standard]>=0.30' 'pydantic>=2.7' "
            "diffusers transformers accelerate safetensors peft sentencepiece "
            "'huggingface_hub[hf_transfer]' "
            # onnxruntime-gpu и onnxruntime делят один модуль: если стоят оба, InsightFace молча
            # уходит на CPU. Ставим только gpu-вариант. Если он окажется собран под чужую CUDA,
            # детект лица уйдёт на CPU — это доли секунды на кадр против секунд на шаг SDXL,
            # то есть терпимо, и подбирать точную версию под драйвер не стоит GPU-времени.
            "insightface onnxruntime-gpu opencv-python-headless 'numpy>=1.26' 'Pillow>=10.3' "
            # Видео: diffusers export_to_video пишет ролик через imageio, а чтение готового mp4
            # для метрики лица идёт через плагин pyav. Без av файл откроется только на запись, и
            # похожесть по среднему кадру молча вернёт None.
            "imageio imageio-ffmpeg av ftfy "

            # Восстановление лиц в ролике. basicsr тянется как зависимость gfpgan и ломается на
            # свежем torchvision (убрали transforms.functional_tensor) — подменяем модуль в
            # worker/video_enhance.py, ставить старый torchvision ради этого нельзя.
            "gfpgan "
            "2>&1 | tail -3"
        ), timeout=2400)
        append_log(pod_id, install.stdout.strip()[-300:] or "pip завершён")

        # Проверяем стек ДО запуска воркера: иначе рассинхрон версий всплывёт как молчаливо
        # неподнявшийся сервис, и разбираться придётся по хвосту worker.log.
        check = _ssh(host, port, (
            "cd /root/app && PYTHONPATH=/root/app python -c "
            "\"import torch,diffusers,transformers;"
            "print('torch',torch.__version__,'cuda',torch.cuda.is_available());"
            "print('diffusers',diffusers.__version__,'transformers',transformers.__version__);"
            "from worker.config import PHOTO_MODEL_CONFIGS;"
            "print('чекпойнтов в конфиге',len(PHOTO_MODEL_CONFIGS))\" 2>&1 | tail -6"
        ), timeout=600)
        append_log(pod_id, check.stdout.strip()[-400:])
        if "cuda True" not in check.stdout:
            raise RuntimeError("стек не собрался: " + check.stdout.strip()[-200:])

        # --- запуск ---
        # Через файл-скрипт, а не одной строкой в ssh: в bash "&" имеет меньший приоритет, чем
        # "&&", поэтому "cd /app && ... &" уводит в фон всю подоболочку. Она наследует stdout
        # ssh-сессии и держит канал открытым — ssh не завершался, развёртывание падало по
        # таймауту, хотя воркер стартовал.
        # Заодно воркер можно перезапустить одной командой при разборе проблем.
        append_log(pod_id, "поднимаю воркер…")
        start_script = "\n".join([
            "#!/usr/bin/env bash",
            "cd /root/app",
            f"export HF_TOKEN={hf_token}",
            f"export WORKER_API_KEY={worker_key}",
            "export HF_HUB_ENABLE_HF_TRANSFER=1",
            f"export DEFAULT_IDENTITY_MODE={record.identity_mode}",
            # Кеш весов на контейнерный диск, не на /workspace: тот сетевой, с квотой 20 GB, и
            # один SDXL с CLIP-энкодером её переполняет ("Disk quota exceeded" уже после аренды).
            # Плата — кеш не переживает удаление пода, но переживает redeploy.
            "export HF_HOME=/root/hf",
            "export MODEL_CACHE_DIR=/root/models",
            # Персонажи и результаты, наоборот, на томе: эмбеддинг эталона невосстановим, потерять
            # его вместе с подом нельзя. Весят они килобайты, в квоту укладываются с запасом.
            "export OUTPUT_DIR=/workspace/output",
            "export CHARACTER_DIR=/workspace/characters",
            "export PYTHONPATH=/root/app",
            "exec python -m runpod_worker.worker --mode api",
        ])
        _ssh(host, port, (
            "cat > /workspace/start_worker.sh <<'WORKER_EOF'\n"
            + start_script
            + "\nWORKER_EOF\nchmod +x /workspace/start_worker.sh"
        ), timeout=120)

        # Старый процесс глушим перед стартом: redeploy на уже работающем поде иначе оставил бы
        # два воркера на одном порту, и запросы уходили бы в старый код.
        _ssh(host, port, "pkill -f 'runpod_worker.worker' || true", timeout=60)
        _ssh(host, port, (
            "setsid nohup /workspace/start_worker.sh > /workspace/worker.log 2>&1 < /dev/null & "
            "echo started"
        ), timeout=120)
        append_log(pod_id, "воркер запущен, жду /health…")

        import time

        # Первый ответ приходит не сразу: uvicorn стартует за секунды, но импорт torch и diffusers
        # на холодном диске занимает до минуты.
        for _ in range(40):
            if worker_health(pod_id):
                update_record(pod_id, deploy_state="ready")
                append_log(pod_id, "готов: /health отвечает")
                return
            time.sleep(15)

        tail = _ssh(host, port, "tail -20 /workspace/worker.log", timeout=60)
        append_log(pod_id, "воркер не ответил. Хвост лога:\n" + tail.stdout.strip()[-600:])
        update_record(pod_id, deploy_state="failed")
    except Exception as exc:  # noqa: BLE001 — в лог пода должно попасть любое падение
        append_log(pod_id, f"ошибка развёртывания: {exc}")
        update_record(pod_id, deploy_state="failed")


def deploy_async(pod_id: str) -> None:
    threading.Thread(target=deploy_worker, args=(pod_id,), daemon=True).start()


def _fetch_over_http(pod_id: str, remote_path: str, local_path) -> bool:
    """Скачивает файл через HTTP-эндпоинт воркера — основной путь для больших файлов.

    scp рвётся на десятках мегабайт: ролики 15-38 MB приходили обрезанными, а mp4 держит
    метаданные в конце файла, поэтому усечённая копия не открывается вовсе.
    """
    from pipeline import prompt_config

    key = prompt_config._get(prompt_config.read_env_file(), "WORKER_API_KEY") or pod_id
    base = worker_base(pod_id, long_running=True)
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with requests.get(f"{base}/file", params={"path": remote_path},
                          headers={"X-Api-Key": key}, stream=True, timeout=1800) as response:
            if response.status_code != 200:
                return False
            with open(local_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    handle.write(chunk)
    except requests.RequestException:
        return False
    return local_path.exists() and local_path.stat().st_size > 0


def fetch_file(pod_id: str, remote_path: str, local_path) -> bool:
    """Скачивает файл с пода.

    Воркер отдаёт путь на диске пода, а не сам файл: гонять картинки через JSON дорого, и до
    конца прогона они всё равно лежат на поде. Забираем только то, что нужно посмотреть.
    """
    record = get_record(pod_id)
    host, port = (record.ssh_host, record.ssh_port) if record else (None, None)
    if not host:
        info = ssh_info(pod_id)
        if not info:
            return False
        host, port = info

    # Сначала HTTP: на файлах больше десятка мегабайт scp обрывается, и mp4 после этого не
    # открывается вовсе. scp остаётся запасным путём — он не зависит от того, жив ли воркер.
    if _fetch_over_http(pod_id, remote_path, local_path):
        return True

    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["scp", *_ssh_opts(), "-P", str(port), f"root@{host}:{remote_path}", str(local_path)],
        capture_output=True, text=True, timeout=600,
    )
    return result.returncode == 0 and local_path.exists()


# --- SSH-туннель ------------------------------------------------------------
# У HTTP-прокси RunPod свой таймаут, короче наших операций: создание персонажа и тем более видео
# в него не укладываются. Симптом обманчивый — вместо ошибки приходит HTML Cloudflare, и клиент
# видит «сломанный JSON», хотя на поде всё досчитывается.
# Туннель убирает прокси из цепочки: запрос идёт на localhost пода через тот же SSH.

_tunnels: dict = {}
_tunnel_lock = threading.Lock()


def open_tunnel(pod_id: str, local_port: int = 0) -> Optional[str]:
    """Поднимает (или переиспользует) SSH-туннель к воркеру и возвращает базовый URL.

    None — если под недоступен по SSH; вызывающий код в этом случае должен откатиться на прокси:
    для коротких запросов (/health, один кадр) его вполне хватает.
    """
    import socket
    import time

    with _tunnel_lock:
        existing = _tunnels.get(pod_id)
        if existing and existing["process"].poll() is None:
            return f"http://127.0.0.1:{existing['port']}"

        record = get_record(pod_id)
        host, port = (record.ssh_host, record.ssh_port) if record else (None, None)
        if not host:
            info = ssh_info(pod_id)
            if not info:
                return None
            host, port = info

        if not local_port:
            # Порт выбирает ОС: подов может быть несколько одновременно, и фиксированный номер
            # означал бы конфликт туннелей между ними.
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                local_port = probe.getsockname()[1]

        process = subprocess.Popen(
            ["ssh", *_ssh_opts(), "-N",
             "-L", f"{local_port}:127.0.0.1:{WORKER_PORT}",
             "-p", str(port), f"root@{host}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

        # Ждём, пока порт начнёт принимать: ssh поднимает форвардинг не мгновенно, и первый же
        # запрос иначе упирается в ConnectionRefused.
        for _ in range(30):
            if process.poll() is not None:
                return None
            try:
                with socket.create_connection(("127.0.0.1", local_port), timeout=1):
                    _tunnels[pod_id] = {"process": process, "port": local_port}
                    return f"http://127.0.0.1:{local_port}"
            except OSError:
                time.sleep(1)

        process.terminate()
        return None


def close_tunnel(pod_id: str) -> None:
    with _tunnel_lock:
        tunnel = _tunnels.pop(pod_id, None)
    if tunnel:
        tunnel["process"].terminate()


def worker_base(pod_id: str, *, long_running: bool = False) -> str:
    """Базовый URL воркера.

    long_running=True — операция заведомо длиннее таймаута прокси (создание персонажа, видео):
    поднимаем туннель. Для коротких запросов прокси проще и не требует живого SSH.
    """
    if long_running:
        tunnelled = open_tunnel(pod_id)
        if tunnelled:
            return tunnelled
    return worker_url(pod_id)
