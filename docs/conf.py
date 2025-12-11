from __future__ import annotations

import sys
from pathlib import Path

# Ensure the package under ``src/`` is importable when building docs locally
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

project = "atomscale"
copyright = "2025, Atomscale"
author = "Atomscale"
release = "2025"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
]

try:
    import sphinx_autodoc_typehints
except ImportError:
    sphinx_autodoc_typehints = None
else:
    extensions.append("sphinx_autodoc_typehints")

## Include Python objects as they appear in source files
## Default: alphabetically ('alphabetical')
autodoc_member_order = "bysource"
## Default flags used by autodoc directives
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "special-members": "__call__",
    "no-index": True,
}
autoclass_content = "both"
# Show type hints in the signature only to avoid duplicate cross references
autodoc_typehints = "none"

# sphinx-autodoc-typehints settings
typehints_fully_qualified = False
always_document_param_types = True
typehints_use_signature = True
typehints_use_signature_return = True
simplify_optional_unions = True
## Generate autodoc stubs with summaries from code
autosummary_generate = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
# html_static_path = ['_static']

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    "github_button": True,
    "github_type": "star&v=2",
    "github_user": "atomic-data-sciences",
    "github_repo": "api-client",
    "github_banner": True,
    "description": "Python SDK",
}

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
html_sidebars = {
    "**": [
        "about.html",
        "navigation.html",
        "searchbox.html",
    ]
}

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "adsdoc"


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (
        master_doc,
        "atomscale",
        "Atomscale API Client Documentation",
        [author],
        1,
    )
]


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "atomscale",
        "Atomscale API Client Documentation",
        author,
        "atomscale",
        "Atomscale API.",
        "",
    ),
]

# -- Extension configuration -------------------------------------------------

autodoc_mock_imports = []

# Example configuration for intersphinx: refer to the Python standard library.
## Add Python version number to the default address to corretcly reference
## the Python standard library
# intersphinx_mapping = {"https://docs.python.org/3.8": None}
