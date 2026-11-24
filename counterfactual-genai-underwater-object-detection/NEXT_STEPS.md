# Continuation and recovery

The full frozen research horizon and final draft review are complete. Start with RESEARCH_RESULTS.md and runs/research_complete.json. The main PDF has 8 pages and 5 figures; the companion study has 4 pages. The result does not support a real-data improvement claim for this generative agent.

To verify or recover work after interruption:

```bash
cd /home/meow/Documents/repos/MVML2027/counterfactual-genai-underwater-object-detection
bash scripts/resume_research.sh
```

The runner checks completed stage signatures and artifact digests. It skips intact stages, resumes unfinished batches/checkpoints and retains complete status only if the final hash-bound review is intact. Never use the earlier --skip-download-if-active handoff switch after power loss. Reconnect the local RTX 5000 Ada if necessary; the UUID guard refuses the smaller GPU. No automatic reboot-start service was installed.

Do not silently modify frozen source/configuration/model/data artifacts or delete the test freeze. The current held-out cohort is already observed. Further method development needs a separate recorded study and fresh evaluation evidence. Editorial changes must preserve prior PDFs/receipts, rebuild the affected LaTeX documents, and renew the affected document reviews and explicit manuscript-stage output supersession.

All project work must remain in this directory. Leave evidence-genAI-underwater-acoustic-reasoning unchanged. The drafts have not been submitted; author metadata, final conference format and supplement submission policy are author-review items.

The editable document sources are `manuscript/main.tex` + `main.bib` and `manuscript/supplementary.tex` + `supplementary.bib`. Rebuild with `bash scripts/build_latex.sh`; this is document compilation only and does not run research experiments. Current reading PDFs are `manuscript/main_paper.pdf` and `manuscript/supplement.pdf`. Every successful build synchronizes them with the validated copies under `manuscript/latex_build/`. The user opens these files directly; do not launch UI previews unless requested. See `manuscript/README.md` for paths, dependencies and review records.
