You are an expert technical writer and senior software engineer.

Task:
Generate (1) a portfolio-style README.md and (2) a longer article.md describing my work on an in-app interactive chart builder feature for a multi-service platform.

Important constraints:
- Do NOT invent facts, metrics, components, or architecture not supported by the provided materials.
- If something is unclear, explicitly mark it as an assumption or omit it.
- Keep everything anonymized.
- Prefer concrete, code-grounded statements.

Inputs:
- CONTEXT.md - context of my work in the whole project I work in
- new_plot_methods/ - the folder with code for new charts which were built from scratch
- my_changes_from_root.patch - the result of git log for the period from the initial version to the current version, which contains only my commits. This is the main file to analyze while building the README.md file. I will not include my_changes_from_root into the final portfolio repository


Output:
1) generated/README.md
- Clearly separate work built from scratch vs transformed legacy
- Include sections: "What is intentionally omitted" and write there why you omitted smth

2) generated/article.md
- 2–4 pages equivalent
- Narrative structure: problem → constraints → solution → trade-offs → outcomes
- Include 2–3 representative Before → After examples based on diffs
- Add a "Lessons learned" section grounded in evidence

Method:
- Internally map claims to supporting lines from the my_changes_from_root.patch
- Write outputs strictly based on evidence

Begin.