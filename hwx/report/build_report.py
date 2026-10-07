#!/usr/bin/env python
"""Build the results report page: collect numbers from results/ and inline them into the template.

    python report/build_report.py          # -> report/build/skin_lesion_study.html (publish this file)
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
data = subprocess.run([sys.executable, str(HERE / "make_report_data.py")], check=True, capture_output=True, text=True).stdout.strip()
page = (HERE / "report_template.html").read_text().replace("__DATA__", data)
out = HERE / "build" / "skin_lesion_study.html"
out.parent.mkdir(exist_ok=True)
out.write_text(page)
print(f"wrote {out} ({len(page) / 1024:.0f} KB)")
