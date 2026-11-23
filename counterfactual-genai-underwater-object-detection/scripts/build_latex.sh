#!/usr/bin/env bash
# Compile standalone LaTeX drafts without touching the existing ReportLab PDFs.
set -euo pipefail

LATEX_PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd -- "$LATEX_PROJECT_ROOT"
# All Python, TeX and rendering scratch files stay in this submission.
source "$LATEX_PROJECT_ROOT/scripts/project_env.sh"
export TEXMFVAR="$LATEX_PROJECT_ROOT/.cache/texmf-var"
export TEXMFCONFIG="$LATEX_PROJECT_ROOT/.cache/texmf-config"
export TEXMFHOME="$LATEX_PROJECT_ROOT/.cache/texmf-home"
export TEXMFOUTPUT="$LATEX_PROJECT_ROOT/manuscript/latex_build"
export TEXINPUTS="$LATEX_PROJECT_ROOT/manuscript//:"
export BIBINPUTS="$LATEX_PROJECT_ROOT/manuscript:"

usage() {
    cat <<'USAGE'
Usage: scripts/build_latex.sh [both|main|supplementary]

Default: build both standalone manuscripts, including BibTeX and page renders.
  main           manuscript/main.tex + manuscript/main.bib; 1-8 pages
  supplementary  manuscript/supplementary.tex + supplementary.bib; 3-4 pages

Validated PDFs: manuscript/latex_build/{main,supplementary}.pdf
Latest renders: manuscript/latex_renders/{main,supplementary}/page-*.png
Every attempt retains its TeX/BibTeX logs and auxiliary files in
manuscript/latex_build/runs/. Failed builds do not replace validated PDFs.
The existing manuscript/main_paper.pdf and supplement.pdf are not touched.
USAGE
}

if (( $# > 1 )); then usage >&2; exit 2; fi
case "${1:-both}" in
    both) LATEX_DOCUMENTS=(main supplementary) ;;
    main) LATEX_DOCUMENTS=(main) ;;
    supplementary) LATEX_DOCUMENTS=(supplementary) ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

LATEX_PYTHON="$LATEX_PROJECT_ROOT/.venv/bin/python"
[[ -x "$LATEX_PYTHON" ]] || { printf 'Missing project Python: %s\n' "$LATEX_PYTHON" >&2; exit 1; }
for LATEX_TOOL in pdflatex bibtex pdftoppm flock rg; do
    command -v "$LATEX_TOOL" >/dev/null || { printf 'Required tool unavailable: %s\n' "$LATEX_TOOL" >&2; exit 1; }
done
for LATEX_DOCUMENT in "${LATEX_DOCUMENTS[@]}"; do
    for LATEX_EXTENSION in tex bib; do
        LATEX_SOURCE="$LATEX_PROJECT_ROOT/manuscript/$LATEX_DOCUMENT.$LATEX_EXTENSION"
        [[ -s "$LATEX_SOURCE" ]] || { printf 'Missing or empty manuscript input: %s\n' "$LATEX_SOURCE" >&2; exit 1; }
    done
done

LATEX_BUILD_ROOT="$LATEX_PROJECT_ROOT/manuscript/latex_build"
LATEX_RENDER_ROOT="$LATEX_PROJECT_ROOT/manuscript/latex_renders"
LATEX_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
LATEX_RUN_ROOT="$LATEX_BUILD_ROOT/runs/$LATEX_RUN_ID"
mkdir -p -- "$TMPDIR" "$TEXMFVAR" "$TEXMFCONFIG" "$TEXMFHOME" "$LATEX_RUN_ROOT" "$LATEX_RENDER_ROOT/history"
exec 9>"$LATEX_BUILD_ROOT/build.lock"
flock -n 9 || { printf 'Another LaTeX build owns %s\n' "$LATEX_BUILD_ROOT/build.lock" >&2; exit 1; }
trap 'LATEX_EXIT_STATUS=$?; if (( LATEX_EXIT_STATUS != 0 )); then printf "Build failed; retained attempt logs: %s\n" "$LATEX_RUN_ROOT" >&2; fi' EXIT

run_logged() {
    local stage_log=$1
    shift
    if "$@" >"$stage_log" 2>&1; then
        return 0
    else
        local stage_status=$?
        printf 'Failed command; log: %s\n' "$stage_log" >&2
        tail -n 70 "$stage_log" >&2
        return "$stage_status"
    fi
}

for LATEX_DOCUMENT in "${LATEX_DOCUMENTS[@]}"; do
    LATEX_DOC_BUILD="$LATEX_RUN_ROOT/$LATEX_DOCUMENT"
    LATEX_DOC_RENDER="$LATEX_RENDER_ROOT/.$LATEX_DOCUMENT.$LATEX_RUN_ID"
    mkdir -p -- "$LATEX_DOC_BUILD" "$LATEX_DOC_RENDER"
    LATEX_SOURCE="$LATEX_PROJECT_ROOT/manuscript/$LATEX_DOCUMENT.tex"
    LATEX_PDF="$LATEX_DOC_BUILD/$LATEX_DOCUMENT.pdf"
    printf 'Building %s; logs: %s\n' "$LATEX_DOCUMENT" "$LATEX_DOC_BUILD"
    cd -- "$LATEX_PROJECT_ROOT/manuscript"
    run_logged "$LATEX_DOC_BUILD/pdflatex-pass1.stdout.log" \
        pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error \
        -file-line-error -recorder -synctex=0 -output-directory="$LATEX_DOC_BUILD" "$LATEX_SOURCE"
    if ! rg -F -q '\bibdata{' "$LATEX_DOC_BUILD/$LATEX_DOCUMENT.aux"; then
        printf 'Standalone source must reference its BibTeX database: %s\n' "$LATEX_SOURCE" >&2
        exit 1
    fi
    cd -- "$LATEX_DOC_BUILD"
    run_logged "$LATEX_DOC_BUILD/bibtex.stdout.log" bibtex "$LATEX_DOCUMENT"
    cd -- "$LATEX_PROJECT_ROOT/manuscript"
    for LATEX_PASS in 2 3; do
        run_logged "$LATEX_DOC_BUILD/pdflatex-pass$LATEX_PASS.stdout.log" \
            pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error \
            -file-line-error -recorder -synctex=0 -output-directory="$LATEX_DOC_BUILD" "$LATEX_SOURCE"
    done

    # Only the final TeX log is judged: initial passes naturally lack citations.
    LATEX_FINAL_LOG="$LATEX_DOC_BUILD/$LATEX_DOCUMENT.log"
    if rg -n -i 'undefined|Overfull \\[hv]box|Label\(s\) may have changed|Rerun to get cross-references right' \
        "$LATEX_FINAL_LOG" >"$LATEX_DOC_BUILD/layout-errors.log"; then
        printf 'Unresolved references/citations, overfull boxes, or unstable labels:\n' >&2
        cat "$LATEX_DOC_BUILD/layout-errors.log" >&2
        exit 1
    fi
    if rg -n -i "Warning--I didn't find|error message|I couldn't open database" \
        "$LATEX_DOC_BUILD/$LATEX_DOCUMENT.blg" >"$LATEX_DOC_BUILD/bibliography-errors.log"; then
        cat "$LATEX_DOC_BUILD/bibliography-errors.log" >&2
        exit 1
    fi

    cd -- "$LATEX_PROJECT_ROOT"
    run_logged "$LATEX_DOC_BUILD/page-validation.stdout.log" \
        "$LATEX_PYTHON" - "$LATEX_DOCUMENT" "$LATEX_PDF" "$LATEX_SOURCE" \
        "$LATEX_PROJECT_ROOT/manuscript/$LATEX_DOCUMENT.bib" "$LATEX_DOC_BUILD/validation.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
from pypdf import PdfReader

name, pdf, source, bib, receipt = sys.argv[1:]
reader = PdfReader(pdf, strict=True)
if reader.is_encrypted:
    raise SystemExit("Encrypted research PDF is not supported")
count = len(reader.pages)
low, high = (1, 8) if name == "main" else (3, 4)
if not low <= count <= high:
    raise SystemExit(f"Invalid {name} page count: {count}; required {low}-{high}")
paths = [pdf, source, bib]
summary = {
    "document": name,
    "pages": count,
    "page_count_bounds": [low, high],
    "sha256": {str(Path(path)): hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in paths},
    "visual_inspection_complete": False,
    "scope": "Compilation and arithmetic checks only; rendered pages still require visual review.",
}
Path(receipt).write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
    run_logged "$LATEX_DOC_BUILD/pdftoppm.stdout.log" \
        pdftoppm -png -r 110 "$LATEX_PDF" "$LATEX_DOC_RENDER/page"
    if [[ -d "$LATEX_RENDER_ROOT/$LATEX_DOCUMENT" ]]; then
        mv -- "$LATEX_RENDER_ROOT/$LATEX_DOCUMENT" "$LATEX_RENDER_ROOT/history/$LATEX_DOCUMENT.$LATEX_RUN_ID.previous"
    fi
    mv -- "$LATEX_DOC_RENDER" "$LATEX_RENDER_ROOT/$LATEX_DOCUMENT"
    # Atomic publication; preserve the attempted PDF and all logs in runs/.
    cp -- "$LATEX_PDF" "$LATEX_BUILD_ROOT/.$LATEX_DOCUMENT.$LATEX_RUN_ID.pdf"
    mv -- "$LATEX_BUILD_ROOT/.$LATEX_DOCUMENT.$LATEX_RUN_ID.pdf" "$LATEX_BUILD_ROOT/$LATEX_DOCUMENT.pdf"
    cp -- "$LATEX_DOC_BUILD/validation.json" "$LATEX_BUILD_ROOT/$LATEX_DOCUMENT.validation.json"
    printf 'Validated PDF: %s\nRendered pages: %s\n' \
        "$LATEX_BUILD_ROOT/$LATEX_DOCUMENT.pdf" "$LATEX_RENDER_ROOT/$LATEX_DOCUMENT"
done
printf 'Build complete. Visual review remains required; all attempt logs: %s\n' "$LATEX_RUN_ROOT"
