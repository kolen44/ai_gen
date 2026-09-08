#!/usr/bin/env bash
# Разворачивает ComfyUI на поде RunPod под две запрошенные заказчиком LoRA:
#   - kp-forks/Flux-uncensored          -> база FLUX.1-dev  (фото)
#   - rzgar/Wan2.2_LightX2V_4Step_Uncensored -> база Wan2.2-I2V-A14B (видео)
#
# Обе ссылки — это LoRA, а не самостоятельные модели, поэтому основной объём скачивания здесь —
# базы (~106 GB), а не сами LoRA (~2.5 GB). Отсюда требование к диску, см. WORKDIR ниже.
#
# Веса кладём на сетевой том (/workspace), а не в контейнерный диск: пересоздание пода не должно
# означать повторное скачивание 106 GB. Скрипт идемпотентен — повторный запуск ничего не качает
# заново (в dl() стоит проверка на существующий файл).
#
# Запуск:  HF_TOKEN=hf_... bash comfyui_uncensored_setup.sh
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace}"
COMFY="$WORKDIR/ComfyUI"
export COMFY HF_TOKEN   # оба читает python-блок в конце скрипта
: "${HF_TOKEN:?HF_TOKEN обязателен: FLUX.1-dev — gated-репозиторий, без токена отдаёт 401}"

log() { echo -e "\n=== $* ===\n"; }

# --- ComfyUI + менеджер нод -------------------------------------------------
if [ ! -d "$COMFY" ]; then
  log "ComfyUI: клонирование"
  git clone --depth 1 https://github.com/comfyanonymous/ComfyUI "$COMFY"
  git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Manager \
      "$COMFY/custom_nodes/ComfyUI-Manager"
fi

log "ComfyUI: зависимости"
# torch в образе runpod/pytorch уже собран под конкретную CUDA — переустанавливать его нельзя:
# "pip install -U torch" тянет билд под другую CUDA и роняет драйвер (тот же грабль, что описан в
# deploy/runpod_setup.md, п.3). Поэтому ставим только requirements ComfyUI и huggingface_hub.
pip install -q -r "$COMFY/requirements.txt"
pip install -q "huggingface_hub[hf_transfer]"
export HF_HUB_ENABLE_HF_TRANSFER=1   # без него 106 GB качаются в разы дольше

# CLI переименовали: в свежем huggingface_hub это `hf`, в версиях постарше — `huggingface-cli`.
# Образ пода может нести любую из них, поэтому определяем один раз, а не гадаем.
if command -v hf >/dev/null 2>&1; then HFCLI=hf; else HFCLI=huggingface-cli; fi

# --- загрузчик --------------------------------------------------------------
# hf download вместо curl: сам ретраит обрывы и докачивает, что на файлах по 28 GB существенно.
dl() { # dl <repo_id> <файл-в-репо> <каталог-назначения> [имя-файла-на-диске]
  local repo="$1" src="$2" dest="$3" name="${4:-$(basename "$2")}"
  mkdir -p "$dest"
  if [ -f "$dest/$name" ]; then echo "  есть: $name"; return; fi
  echo "  качаю: $repo/$src -> $name"
  "$HFCLI" download "$repo" "$src" --local-dir "$dest/.staging" --token "$HF_TOKEN"
  mv "$dest/.staging/$src" "$dest/$name"
  rm -rf "$dest/.staging"
}

M="$COMFY/models"

# --- FLUX.1-dev (база под Flux-uncensored) ----------------------------------
# Транс­формер берём из официального BFL-репозитория; текстовые энкодеры и VAE — отдельными
# файлами, потому что ComfyUI грузит Flux через UNETLoader + DualCLIPLoader, а не как единый
# чекпойнт. t5xxl берём в fp16: на 80 GB VRAM экономить на энкодере незачем, а fp8 заметно
# режет следование промпту.
log "FLUX.1-dev (~34 GB)"
dl black-forest-labs/FLUX.1-dev  flux1-dev.safetensors  "$M/diffusion_models"
dl black-forest-labs/FLUX.1-dev  ae.safetensors         "$M/vae"  flux_ae.safetensors
dl comfyanonymous/flux_text_encoders  clip_l.safetensors        "$M/text_encoders"
dl comfyanonymous/flux_text_encoders  t5xxl_fp16.safetensors    "$M/text_encoders"

# --- Wan2.2 I2V A14B (база под LightX2V-LoRA) -------------------------------
# A14B — MoE из двух экспертов (high/low noise), поэтому и файлов два, и LoRA в репозитории rzgar
# тоже две: каждая вешается на своего эксперта. К плотной Wan2.2-TI2V-5B эта пара не применяется
# в принципе — вот почему под эту задачу нельзя переиспользовать 5B из worker/config.py.
#
# VAE — именно wan_2.1_vae: A14B унаследовал VAE от Wan 2.1. Файл wan2.2_vae.safetensors
# относится к TI2V-5B и с A14B даст мусор на выходе.
log "Wan2.2-I2V-A14B (~69 GB)"
R=Comfy-Org/Wan_2.2_ComfyUI_Repackaged
dl "$R" split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors "$M/diffusion_models"
dl "$R" split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors  "$M/diffusion_models"
dl "$R" split_files/text_encoders/umt5_xxl_fp16.safetensors "$M/text_encoders"
dl "$R" split_files/vae/wan_2.1_vae.safetensors             "$M/vae"

# --- собственно запрошенные LoRA --------------------------------------------
log "LoRA заказчика (~2.5 GB)"
dl rzgar/Wan2.2_LightX2V_4Step_Uncensored Wan2.2_LightX2V_high_n54vv.safetensors "$M/loras"
dl rzgar/Wan2.2_LightX2V_4Step_Uncensored Wan2.2_LightX2V_low_n54vv.safetensors  "$M/loras"

# У kp-forks/Flux-uncensored имя файла LoRA в репозитории заранее не зафиксировано (репозиторий
# форкнутый, автор мог переименовать), поэтому находим .safetensors через API, а не хардкодим.
python - <<'PY'
import os, json, urllib.request, pathlib, shutil
from huggingface_hub import hf_hub_download
tok = os.environ["HF_TOKEN"]; repo = "kp-forks/Flux-uncensored"
req = urllib.request.Request(f"https://huggingface.co/api/models/{repo}",
                             headers={"Authorization": f"Bearer {tok}"})
files = [s["rfilename"] for s in json.load(urllib.request.urlopen(req))["siblings"]
         if s["rfilename"].endswith(".safetensors")]
if not files:
    raise SystemExit(f"в {repo} нет .safetensors — проверьте репозиторий вручную")
dest = pathlib.Path(os.environ["COMFY"]) / "models" / "loras" / "flux_uncensored.safetensors"
if dest.exists():
    print("  есть: flux_uncensored.safetensors")
else:
    print(f"  качаю: {repo}/{files[0]}")
    shutil.copy(hf_hub_download(repo, files[0], token=tok), dest)
PY

log "Готово. Веса:"
du -sh "$M"/diffusion_models "$M"/text_encoders "$M"/vae "$M"/loras

log "Запуск ComfyUI на 0.0.0.0:8188"
cd "$COMFY"
exec python main.py --listen 0.0.0.0 --port 8188
