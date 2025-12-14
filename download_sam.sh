#!/usr/bin/env bash
set -euo pipefail

LINKS_FILE="${1:-sam.txt}"                       # input file: "sa_000123.tar  https://..."
TAR_DIR="${2:-data/ShareGPT4V/data/sam/raw}"         # destination dir for .tar files
EXTRACT_DIR="${3:-data/ShareGPT4V/data/sam/images}"  # destination dir for extracted images
mkdir -p "$TAR_DIR"
mkdir -p "$EXTRACT_DIR"

# Download tar files
while read -r name url _; do
  # skip blank/header lines
  [[ -z "${name:-}" || -z "${url:-}" ]] && continue
  [[ "${url}" != http* ]] && continue
  [[ "$name" != sa_[0-9][0-9][0-9][0-9][0-9][0-9].tar ]] && continue

  # numeric index from sa_XXXXXX.tar -> only 0..50
  n=$((10#${name:3:6}))
  if (( n >= 0 && n <= 50 )); then
    dest="$TAR_DIR/$name"
    if [[ -s "$dest" ]]; then
      echo "Skipping download of $name (already exists)"
    else
      echo "Downloading $name"
      curl -L --fail --continue-at - -o "$dest" "$url"
    fi
  fi
done < "$LINKS_FILE"


# Extract tar files
echo ""
echo "Extracting images..."
extracted_count=0
for f in "$TAR_DIR"/sa_*.tar; do
  if [[ ! -f "$f" ]]; then
    continue
  fi

  name=$(basename "$f" .tar)
  echo "Extracting $name..."
  
  # Extract directly to extract directory
  # SAM tar files contain images, extract preserving directory structure if any
  tar -xf "$f" -C "$EXTRACT_DIR" 2>/dev/null || {
    echo "  Warning: Failed to extract $name"
    continue
  }
  
  ((extracted_count++))
done

echo ""
echo "Done! Extracted $extracted_count tar files to: $EXTRACT_DIR"
if [[ -d "$EXTRACT_DIR" ]]; then
  image_count=$(find "$EXTRACT_DIR" -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" -o -name "*.JPG" -o -name "*.PNG" -o -name "*.JPEG" \) 2>/dev/null | wc -l)
  echo "Total images found: $image_count"
fi
