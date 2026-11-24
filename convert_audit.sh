#!/bin/bash
set -e

VENV=".venv/Scripts/python.exe"

# Convert find_h with --max-turns 8
for file in sandbox-gym/data/collect/r5-findh-r3-s1.jsonl sandbox-gym/data/collect/r5-findh-r3-wide.jsonl; do
  outfile="sft/data/_audit_$(basename $file)"
  echo "Converting $file..."
  $VENV sft/from_gym.py "$file" \
    --source "sandbox-gym:find_h" \
    --out "$outfile" \
    --max-turns 8
done

# Convert edit
for file in sandbox-gym/data/collect/r5-edit-r3-s1.jsonl sandbox-gym/data/collect/r5-edit-r3-wide.jsonl; do
  outfile="sft/data/_audit_$(basename $file)"
  echo "Converting $file..."
  $VENV sft/from_gym.py "$file" \
    --source "sandbox-gym:edit" \
    --out "$outfile"
done

# Convert fix with --strip-hint
file=sandbox-gym/data/collect/r5-fix-r3-hint.jsonl
outfile="sft/data/_audit_$(basename $file)"
echo "Converting $file with hint stripping..."
$VENV sft/from_gym.py "$file" \
  --source "sandbox-gym:fix" \
  --out "$outfile" \
  --strip-hint "The project is small: run the tests first, then read only the files the failure points to."

echo "Done converting. Converted files:"
ls -la sft/data/_audit_r5*.jsonl
