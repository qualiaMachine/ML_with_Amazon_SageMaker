import os
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell

# Paths
episodes_dir = "episodes"
notebooks_dir = "notebooks"

# Ensure notebooks directory exists
os.makedirs(notebooks_dir, exist_ok=True)

# Convert each Markdown file in episodes/
for filename in os.listdir(episodes_dir):
    if filename.endswith(".md"):
        md_path = os.path.join(episodes_dir, filename)
        ipynb_path = os.path.join(notebooks_dir, filename.replace(".md", ".ipynb"))

        # Read Markdown content
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        # Create Jupyter notebook
        nb = new_notebook(cells=[new_markdown_cell(md_content)])

        # Save as .ipynb
        with open(ipynb_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

print("Conversion complete!")
