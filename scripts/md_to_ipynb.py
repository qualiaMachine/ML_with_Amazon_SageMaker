import os
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell

# Paths
episodes_dir = "episodes"
notebooks_dir = "notebooks"

# List of Markdown files to ignore (without conversion)
ignore_list = ["SageMaker-overview.md", "Data-storage-setting-up-S3.md", "SageMaker-notebooks-as-controllers.md", "Resource-management-cleanup.md"]  # Add files that shouldn't be converted

# Ensure notebooks directory exists
os.makedirs(notebooks_dir, exist_ok=True)

# Convert each Markdown file in episodes/
for filename in os.listdir(episodes_dir):
    if filename.endswith(".md") and filename not in ignore_list:
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

print("Conversion complete! Excluded:", ", ".join(ignore_list))
