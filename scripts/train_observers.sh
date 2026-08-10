#!/usr/bin/env bash
# Train the four Observation Agents and write their reports on the training split.
#
# Three phases per role: prepare the SFT dataset, train it, then collect the final
# checkpoint into one adapter root. Inference over the training split runs last,
# once every role is trained, and produces the reports the Judge datasets need.
#
# Override any setting from the environment, for example:
#
#   MODEL=checkpoints/Qwen2.5-VL-3B-Instruct GPUS=0,1 bash scripts/train_observers.sh
#   ROLES="texture lighting" bash scripts/train_observers.sh
#   SKIP_INFERENCE=1 bash scripts/train_observers.sh
#
# Arguments after -- are forwarded to ms-swift during training.
set -euo pipefail

MODEL="${MODEL:-checkpoints/Qwen2.5-VL-7B-Instruct}"
DATASET_ROOT="${DATASET_ROOT:-data/FaceVid-Forensics-100K}"
OBSERVATIONS="${OBSERVATIONS:-${DATASET_ROOT}/perception/aggregated/train.jsonl}"
MANIFEST="${MANIFEST:-${DATASET_ROOT}/manifests/train.json}"
DERIVED_DIR="${DERIVED_DIR:-data/derived}"
GPUS="${GPUS:-0,1,2,3}"
ROLES="${ROLES:-texture lighting motion physics}"
SKIP_INFERENCE="${SKIP_INFERENCE:-0}"

# Runs are grouped by base model so a scale study never overwrites another size.
# The -Instruct suffix carries no information here, so it is dropped.
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/$(basename "${MODEL}" | sed 's/-Instruct$//')}"
LORA_ROOT="${LORA_ROOT:-${OUTPUT_ROOT}/shared/lora}"

for path in "${DATASET_ROOT}" "${OBSERVATIONS}"; do
  if [[ ! -e "${path}" ]]; then
    echo "error: ${path} does not exist" >&2
    exit 1
  fi
done

echo "model: ${MODEL}"

for role in ${ROLES}; do
  dataset="${DERIVED_DIR}/observer_${role}.jsonl"
  output="${OUTPUT_ROOT}/sft_observer_${role}"

  echo "=== ${role}: preparing ${dataset} ==="
  python -m src.prepare observer --role "${role}" \
    --source "${OBSERVATIONS}" \
    --dataset-root "${DATASET_ROOT}" \
    --output "${dataset}"

  echo "=== ${role}: training into ${output} ==="
  python -m src.train sft --model "${MODEL}" \
    --role "${role}" \
    --dataset "${dataset}" \
    --output "${output}" \
    --gpus "${GPUS}" \
    "$@"

  # ms-swift writes <output>/checkpoint-N/; flatten the last one so that
  # src.argus_infer --observer-lora-root finds <root>/<role>/adapter_config.json.
  latest="$(find "${output}" -maxdepth 1 -type d -name 'checkpoint-*' \
    | sort -V | tail -1)"
  if [[ -z "${latest}" ]]; then
    echo "error: no checkpoint-* directory under ${output}" >&2
    exit 1
  fi
  echo "=== ${role}: collecting ${latest} -> ${LORA_ROOT}/${role} ==="
  mkdir -p "${LORA_ROOT}/${role}"
  cp "${latest}"/adapter_* "${LORA_ROOT}/${role}/"
  if [[ -f "${latest}/additional_config.json" ]]; then
    cp "${latest}/additional_config.json" "${LORA_ROOT}/${role}/"
  fi
done

if [[ "${SKIP_INFERENCE}" != "0" ]]; then
  echo "done: trained ${ROLES// /, } into ${LORA_ROOT}/ (inference skipped)"
  exit 0
fi

for role in ${ROLES}; do
  report="${DERIVED_DIR}/train_${role}.jsonl"
  echo "=== ${role}: writing ${report} ==="
  python -m src.argus_infer observer --role "${role}" \
    --input "${MANIFEST}" \
    --dataset-root "${DATASET_ROOT}" \
    --base-model "${MODEL}" \
    --observer-lora-root "${LORA_ROOT}" \
    --output "${report}"
done

echo "done: ${ROLES// /, } trained into ${LORA_ROOT}/ and reported into ${DERIVED_DIR}/"
