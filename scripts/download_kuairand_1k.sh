#!/usr/bin/env bash
set -euo pipefail

data_root="${1:-data/raw}"
archive_path="${data_root}/KuaiRand-1K.tar.gz"
dataset_dir="${data_root}/KuaiRand-1K"
download_url="https://zenodo.org/records/10439422/files/KuaiRand-1K.tar.gz"
expected_md5="6b0b9c8222d67fcd4c676218edca3f1f"

mkdir -p "${data_root}"

if [[ -d "${dataset_dir}/data" ]]; then
  echo "KuaiRand-1K is already extracted at ${dataset_dir}"
  exit 0
fi

checksum() {
  if command -v md5 >/dev/null 2>&1; then
    md5 -q "$1"
  elif command -v md5sum >/dev/null 2>&1; then
    md5sum "$1" | awk '{print $1}'
  else
    echo "Neither md5 nor md5sum is available; refusing to extract an unverified archive." >&2
    exit 1
  fi
}

actual_md5=""
if [[ -f "${archive_path}" ]]; then
  actual_md5="$(checksum "${archive_path}")"
fi

if [[ "${actual_md5}" != "${expected_md5}" ]]; then
  curl -L --fail --continue-at - --output "${archive_path}" "${download_url}"
  actual_md5="$(checksum "${archive_path}")"
fi

if [[ "${actual_md5}" != "${expected_md5}" ]]; then
  echo "Checksum mismatch for ${archive_path}" >&2
  echo "expected: ${expected_md5}" >&2
  echo "actual:   ${actual_md5}" >&2
  exit 1
fi

tar -xzf "${archive_path}" -C "${data_root}"
echo "KuaiRand-1K extracted to ${dataset_dir}"
