# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'FedERA'
copyright = '2023, KLIV'
author = 'KLIV'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'autoapi.extension',  # this one is really important
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.autosectionlabel',  # allows referring sections its title, affects `ref`
    'sphinx_design',
    'sphinxcontrib.bibtex',
]
# for 'sphinxcontrib.bibtex' extension
bibtex_bibfiles = ['refer.bib']
bibtex_default_style = 'unsrt'

autodoc_mock_imports = ["numpy", "torch", "torchvision", "pandas"]
autoclass_content = 'both'

templates_path = ['_templates']
# configuration for 'autoapi.extension'
autoapi_type = 'python'
autoapi_dirs = ['../..']
autoapi_template_dir = '_autoapi_temp'
add_module_names = False  # makes Sphinx render package.module.Class as Class

# Add more mapping for 'sphinx.ext.intersphinx'
intersphinx_mapping = {'python': ('https://docs.python.org/3', None),
                       'PyTorch': ('http://pytorch.org/docs/master/', None),
                       'numpy': ('https://numpy.org/doc/stable/', None),
                       'pandas': ('https://pandas.pydata.org/pandas-docs/dev/', None)}

# autosectionlabel throws warnings if section names are duplicated.
# The following tells autosectionlabel to not throw a warning for
# duplicated section names that are in different documents.
autosectionlabel_prefix_document = True

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# Config for 'sphinx.ext.todo'
todo_include_todos = True

# multi-language docs
language = 'en'
locale_dirs = ['../locales/']  # path is example but recommended.
gettext_compact = False  # optional.
gettext_uuid = True  # optional.

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
