---
name: read-arxiv-paper
description: Use this skill when asked to read an arxiv paper given an arxiv URL
argument-hint: <arxiv-url>
---

You will be given a URL of an arxiv paper, for example:
https://www.arxiv.org/abs/2601.07372

### Part 1: Normalize the URL
Rewrite the URL to point to the TeX source (not the PDF) by replacing `/abs/` with `/src/`:
`https://www.arxiv.org/src/2601.07372`

### Part 2: Download the paper source
Fetch the URL and save it as a local `.tar.gz` file at:
`$TMPDIR/arxiv/{arxiv_id}.tar.gz`
If the file already exists, skip this step.

### Part 3: Unpack the archive
Extract the contents into:
`$TMPDIR/arxiv/{arxiv_id}/`

### Part 4: Locate the entrypoint
Find the main `.tex` file — typically named `main.tex`, but it could be anything. Look for the file that contains `\documentclass` as a reliable signal.

### Part 5: Read the paper
Starting from the entrypoint, read the full content of the tex file(s), following any `\input{}` or `\include{}` references to pull in other source files. Read everything up to and including the **Conclusion** section — you can stop before any Appendices, Acknowledgements, or References.

### Part 6: Inspect key figures
While reading, note any important figures referenced in the text. Locate the corresponding image files in the unpacked directory (typically `.png`, `.pdf`, or `.eps` files inside a `figures/` or `imgs/` folder) and view them directly to understand what they show.

Once you've read the paper and inspected the figures, let the user know you're ready for questions.
(base) mo@Mohammeds-Ma