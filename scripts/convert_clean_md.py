import re
from pathlib import Path

input_file = "outputs/50G/parabola_fits/README_parabola.md"
output_file = "outputs/50G/parabola_fits/README_parabola_cleaned.md"

text = Path(input_file).read_text(encoding="utf-8")

# =====================================================
# STEP 1: Convert block equations
#
# [
# equation
# ]
#
# ->
#
# $$
# equation
# $$
# =====================================================

block_pattern = re.compile(
    r'^\[\s*\n(.*?)\n\]$',
    re.MULTILINE | re.DOTALL
)

text = block_pattern.sub(
    lambda m: "$$\n" + m.group(1).strip() + "\n$$",
    text
)

# =====================================================
# STEP 2: Protect all $$ ... $$ blocks
# =====================================================

protected_blocks = []

def protect_block(match):
    protected_blocks.append(match.group(0))
    return f"__BLOCK_{len(protected_blocks)-1}__"

text = re.sub(
    r'\$\$(.*?)\$\$',
    protect_block,
    text,
    flags=re.DOTALL
)

# =====================================================
# STEP 3: Convert inline equations ONLY outside blocks
#
# Converts:
#   (R_n=i) -> $R_n=i$
#
# Avoids ordinary English parentheses as much as possible
# =====================================================

inline_pattern = re.compile(
    r'\(([^()\n]*?[\\_^=+\-/*{}\d][^()\n]*?)\)'
)

text = inline_pattern.sub(
    lambda m: f"${m.group(1)}$",
    text
)

# =====================================================
# STEP 4: Restore protected $$ blocks
# =====================================================

for i, block in enumerate(protected_blocks):
    text = text.replace(f"__BLOCK_{i}__", block)

# =====================================================
# Write output
# =====================================================

Path(output_file).write_text(text, encoding="utf-8")

print("Conversion complete.")