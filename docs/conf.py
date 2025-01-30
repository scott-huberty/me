# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Scott Huberty'
copyright = '2024, Scott Huberty'
author = 'Scott Huberty'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx_gallery.gen_gallery',
    ]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'shibuya'
html_static_path = ['_static']

html_theme_options = {
    "light_logo": "_static/logo.svg",
    "dark_logo": "_static/logo_dark.png",
    "color_mode": "light",
    "accent_color": "tomato",
    "github_url": "https://github.com/scott-huberty",
    "linkedin_url": "https://github.com/scott-huberty",
    "nav_links": [
        {
            "title": "Blog",
            "url": "./auto_examples/index",
        },
        {
            "title": "Portfolio",
            "url": "./portfolio",
        }
    ]
}

html_css_files = [
    "custom.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.1/css/all.min.css"
]

# -- Options for sphinx_gallery ----------------------------------------------
# https://sphinx-gallery.github.io/stable/configuration.html

sphinx_gallery_conf = {
     'examples_dirs': '../examples',   # path to your example scripts
     'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
}