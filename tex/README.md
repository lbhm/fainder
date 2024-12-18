# Latex Source Code

This folder contains the LaTeX source for the paper **"Fainder: A Fast and Accurate Index for Distribution-Aware Dataset Search"**.

## Building

Building the project is tested in two setups.

### Overleaf

This works out of the box.

### Local

To compile the project locally, you need:

- TexLive 2022 or newer
- `biber` on your PATH
- `latexmk` on your PATH

On Ubuntu, run the following command to install all dependencies:

```bash
sudo apt install --no-install-recommends biber latexmk texlive texlive-bibtex-extra texlive-fonts-extra texlive-latex-extra texlive-plain-generic texlive-publishers texlive-science
```

Building the project is tested with the LaTex Workshop plugin for VS Code or `latexmk`.
With `latexmk` execute:

```bash
latexmk -pdf
```

Other tools or manually executing `pdflatex` and `biber` might work as well.
