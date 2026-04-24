"""Convert gran_upgrade_journey.md to PDF via markdown -> HTML -> headless Chrome."""
import subprocess
from pathlib import Path
import markdown
import re

DOCS_DIR = Path(__file__).parent
MD_PATH = DOCS_DIR / "gran_upgrade_journey.md"
HTML_PATH = DOCS_DIR / "gran_upgrade_journey.html"
PDF_PATH = DOCS_DIR / "gran_upgrade_journey.pdf"

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

CSS = """
<style>
body { font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
       max-width: 820px; margin: 40px auto; color: #222; line-height: 1.55;
       padding: 0 24px; }
h1 { border-bottom: 2px solid #333; padding-bottom: 6px; }
h2 { border-bottom: 1px solid #ccc; padding-bottom: 4px; margin-top: 32px; }
h3 { margin-top: 24px; }
code { background: #f4f4f4; padding: 2px 5px; border-radius: 3px;
       font-family: Consolas, Menlo, monospace; font-size: 0.92em; }
pre { background: #f4f4f4; padding: 12px; border-radius: 4px;
      overflow-x: auto; font-size: 0.88em; line-height: 1.4; }
pre code { background: none; padding: 0; }
blockquote { border-left: 4px solid #bbb; color: #555;
             margin: 16px 0; padding: 8px 16px; background: #fafafa; }
table { border-collapse: collapse; margin: 12px 0; font-size: 0.92em;
        width: 100%; }
th, td { border: 1px solid #ccc; padding: 8px 10px; text-align: left;
         vertical-align: top; }
th { background: #f0f0f0; }
img { max-width: 100%; height: auto; display: block;
      margin: 8px auto; border: 1px solid #eee; }
a { color: #0366d6; text-decoration: none; }
a:hover { text-decoration: underline; }
hr { border: none; border-top: 1px solid #ddd; margin: 28px 0; }
</style>
"""

md_text = MD_PATH.read_text(encoding="utf-8")

# Resolve relative image paths to absolute file:// URIs so Chrome can load them.
def fix_img(match):
    alt, rel = match.group(1), match.group(2)
    if rel.startswith(("http://", "https://", "file://")):
        return match.group(0)
    abs_path = (DOCS_DIR / rel).resolve()
    return f"![{alt}]({abs_path.as_uri()})"

md_text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", fix_img, md_text)

html_body = markdown.markdown(
    md_text,
    extensions=["tables", "fenced_code", "toc"],
)

html_full = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>GRAN to DiGress: Upgrade Journey</title>
{CSS}
</head><body>
{html_body}
</body></html>
"""

HTML_PATH.write_text(html_full, encoding="utf-8")
print(f"[html] {HTML_PATH}")

cmd = [
    CHROME,
    "--headless=new",
    "--disable-gpu",
    "--no-sandbox",
    f"--print-to-pdf={PDF_PATH}",
    "--print-to-pdf-no-header",
    HTML_PATH.as_uri(),
]
subprocess.run(cmd, check=True)
print(f"[pdf]  {PDF_PATH}")
