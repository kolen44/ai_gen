#!/usr/bin/env bash
# Разворачивает AnyID (identity-preserving video) на поде RunPod.
#
# AnyID — LoRA (rank/alpha 256) к Wan-AI/Wan2.2-TI2V-5B, то есть к той же плотной 5B-модели,
# которая уже стоит дефолтом в worker/config.py (DEFAULT_VIDEO_MODEL=wan2.2-ti2v-5b). Отсюда
# скромные требования к диску и GPU по сравнению с A14B-веткой.
#
# Почему отдельный скрипт, а не запись в VIDEO_MODEL_CONFIGS: AnyID нельзя подключить через
# diffusers load_lora_weights. Его LoRA нужно ПРЕДВАРИТЕЛЬНО вмерджить в DiT их собственным
# скриптом (trainer/models/wan22/merge_lora.py), а инференс идёт через их generate.py, который
# принимает JSON с путями к референсам. Это чужой пайплайн целиком, а не веса к нашему.
#
# Лицензия: CC BY-NC-SA 4.0 — некоммерческая И share-alike (производные обязаны быть под той же
# лицензией). Жёстче, чем у всего остального в этом проекте. Перед коммерческим использованием
# читать LICENSE в репозитории, а не этот комментарий.
#
# Запуск:  HF_TOKEN=hf_... bash anyid_video_setup.sh
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace}"
REPO="$WORKDIR/AnyID"
: "${HF_TOKEN:?HF_TOKEN обязателен}"

log() { echo -e "\n=== $* ===\n"; }

if [ ! -d "$REPO" ]; then
  log "AnyID: клонирование"
  git clone --depth 1 https://github.com/JoHnneyWang/AnyID "$REPO"
fi

log "Зависимости"
# torch НЕ трогаем — он в образе собран под конкретную CUDA, и "pip install -U torch" тянет билд
# под другую и роняет драйвер (см. deploy/runpod_setup.md, п.3). Их requirements.txt содержит
# torch; ставим с --no-deps-подходом: сначала всё, что не torch.
grep -viE '^torch(vision|audio)?([=<>~!]|$)' "$REPO/requirements.txt" > /tmp/anyid-req.txt
pip install -q -r /tmp/anyid-req.txt
pip install -q "huggingface_hub[hf_transfer]" peft
export HF_HUB_ENABLE_HF_TRANSFER=1
if command -v hf >/dev/null 2>&1; then HFCLI=hf; else HFCLI=huggingface-cli; fi

# --- база и LoRA ---
BASE="$WORKDIR/models/Wan2.2-TI2V-5B"
LORA="$WORKDIR/models/anyid"

log "Wan2.2-TI2V-5B (база, ~10 GB)"
# Не *-Diffusers-вариант: AnyID работает с оригинальной раскладкой репозитория Wan-AI
# (T5-энкодер, VAE и DiT отдельными файлами), а не с diffusers-структурой подпапок. Наш
# worker/config.py ссылается на Diffusers-вариант — это НЕ одно и то же, две разные раскладки
# одних и тех же весов, и подменять одну другой нельзя.
[ -d "$BASE" ] || "$HFCLI" download Wan-AI/Wan2.2-TI2V-5B --local-dir "$BASE" --token "$HF_TOKEN"

log "AnyID LoRA (~1.5 GB)"
[ -d "$LORA" ] || "$HFCLI" download JonneyWang/AnyID --local-dir "$LORA" --token "$HF_TOKEN"

# --- вмердж LoRA в DiT ---
MERGED="$WORKDIR/models/Wan2.2-TI2V-5B-AnyID"
if [ ! -d "$MERGED" ]; then
  log "Вмердж LoRA в DiT (одноразовая операция, результат переиспользуется)"
  cd "$REPO"
  python trainer/models/wan22/merge_lora.py \
      --base_model_path "$BASE" \
      --lora_path "$LORA/anyid_lora.safetensors" \
      --output_path "$MERGED"
else
  echo "  уже вмерджено: $MERGED"
fi

log "Готово"
cat <<'USAGE'
Инференс (референсы — до 5 фото одного человека ИЛИ 1 видео):

  cd /workspace/AnyID
  # prompts/*.json задаёт пары "пути к референсам" + текст сцены; свой файл делать по образцу
  # из репозитория, а не изобретать формат.
  bash scripts/infer.sh

Источник референсов для нашего пайплайна: те же кадры Stage 0, что идут в FaceID portrait
(output/<character>/ — фронтальный плюс ракурсы). Файлы, не эмбеддинги: AnyID принимает
изображения и считает identity сам.
USAGE
