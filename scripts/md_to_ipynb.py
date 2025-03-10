import os
import nbformat
import re
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

# Paths
episodes_dir = "episodes"
notebooks_dir = "notebooks"

# List of Markdown files to ignore (no conversion needed)
ignore_list = [
    "SageMaker-overview.md",
    "Data-storage-setting-up-S3.md",
    "SageMaker-notebooks-as-controllers.md",
    "Resource-management-cleanup.md"
]

# Ensure notebooks directory exists
os.makedirs(notebooks_dir, exist_ok=True)

# Regular expression to detect code blocks (matches ```language\n...\n```)
code_block_pattern = re.compile(r"```(\w+)?\n(.*?)\n```", re.DOTALL)

def split_markdown(md_content):
    """Splits Markdown content into separate Markdown and Code cells."""
    cells = []
    position = 0

    for match in code_block_pattern.finditer(md_content):
        # Extract text before the code block as Markdown
        before_code = md_content[position:match.start()].strip()
        if before_code:
            cells.append(new_markdown_cell(before_code))
        
        # Extract code block content
        code_content = match.group(2).strip()
        if code_content:
            cells.append(new_code_cell(code_content))

        position = match.end()

    # Add any remaining Markdown content after the last code block
    remaining_md = md_content[position:].strip()
    if remaining_md:
        cells.append(new_markdown_cell(remaining_md))
    
    return cells

# Convert each Markdown file in episodes/
for filename in os.listdir(episodes_dir):
    if filename.endswith(".md") and filename not in ignore_list:
        md_path = os.path.join(episodes_dir, filename)
        ipynb_path = os.path.join(notebooks_dir, filename.replace(".md", ".ipynb"))

        # Read Markdown content
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        # Split into Markdown and Code cells
        notebook_cells = split_markdown(md_content)

        # Create Jupyter notebook
        nb = new_notebook(cells=notebook_cells)

        # Save as .ipynb
        with open(ipynb_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

print("Conversion complete! Excluded:", ", ".join(ignore_list))
