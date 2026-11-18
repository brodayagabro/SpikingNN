"""Build REPORT.pdf from REPORT.md using Pandoc and LuaLaTeX.

Run from any directory: python build_pdf.py [--pandoc /path/to/pandoc].
Only typesetting is performed; figures, data and simulations are not rebuilt.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


HERE = Path(__file__).resolve().parent


def main():
    """Typeset the current Markdown report and appendix, then verify PDF text and layout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pandoc", default="pandoc")
    args = parser.parse_args()
    for executable in (args.pandoc, "lualatex", "pdfinfo", "pdftotext"):
        if not shutil.which(executable):
            raise SystemExit(f"Required executable missing: {executable}")
    build = HERE / "pdf_build"
    build.mkdir(exist_ok=True)
    # Pandoc needs native heading IDs; HTML anchors without a following blank
    # line otherwise cause the following Markdown heading to become plain text.
    source = (HERE / "REPORT.md").read_text()
    printable = re.sub(r'<a id="(sec-\d+)"></a>\n(## [^\n]+)',
                       r'\2 {#\1}', source)
    printable = printable.replace('(APPENDIX_PARAMETERS.md)', '(#appendix-parameters)')
    printable += '\n\n\\newpage\n\n' + (HERE / 'APPENDIX_PARAMETERS.md').read_text()
    (build / "REPORT.print.pandoc").write_text(printable)
    (build / "REPORT.print.md").unlink(missing_ok=True)
    env = os.environ.copy()
    env["TEXMFVAR"] = str(build / "texmf-var")
    env["TEXMFCACHE"] = str(build / "texmf-cache")
    env["TEXMFOUTPUT"] = str(build)
    commands = [[
        args.pandoc, "pdf_build/REPORT.print.pandoc", "--standalone",
        "--from=markdown-implicit_figures", "--to=latex",
        "--lua-filter=pdf_format.lua", "--include-in-header=pdf_header.tex",
        "--variable=mainfont:Liberation Serif",
        "--variable=sansfont:DejaVu Sans",
        "--variable=monofont:DejaVu Sans Mono",
        "--variable=monofontoptions:Scale=0.8",
        "--variable=lang:ru-RU", "--variable=fontsize:11pt",
        "--variable=papersize:a4", "--variable=geometry:margin=20mm",
        "--variable=colorlinks:true", "--output=pdf_build/REPORT.tex",
    ]]
    latex = ["lualatex", "-no-shell-escape", "-interaction=nonstopmode",
             "-halt-on-error", "-file-line-error",
             "-output-directory=pdf_build", "pdf_build/REPORT.tex"]
    commands.extend([latex] * 3)
    for index, command in enumerate(commands):
        result = subprocess.run(command, cwd=HERE, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (build / f"pass_{index}.txt").write_text(result.stdout)
        if result.returncode:
            raise SystemExit(result.stdout[-7000:])
    log = (build / "REPORT.log").read_text()
    issues = [line for line in log.splitlines() if any(word in line for word in
              ("Overfull", "Missing character", "undefined", "LaTeX Error"))]
    (build / "layout_warnings.txt").write_text("\n".join(issues) + "\n")
    if issues:
        raise SystemExit("PDF layout needs attention:\n" + "\n".join(issues))
    pdf = build / "REPORT.pdf"
    text = subprocess.check_output(["pdftotext", "-layout", str(pdf), "-"], text=True)
    (build / "REPORT.txt").write_text(text)
    if "Афферентное управление" not in text or "6. Выводы" not in text:
        raise SystemExit("PDF text validation failed: title or final section missing")
    info = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
    pages = int(re.search(r"Pages:\s+(\d+)", info).group(1))
    shutil.copy2(pdf, HERE / "REPORT.pdf")
    summary = {
        "source": "REPORT.md", "pdf": "REPORT.pdf", "pages": pages,
        "source_sha256": hashlib.sha256((HERE / "REPORT.md").read_bytes()).hexdigest(),
        "appendix_sha256": hashlib.sha256((HERE / "APPENDIX_PARAMETERS.md").read_bytes()).hexdigest(),
        "layout_warnings": issues,
    }
    (build / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
