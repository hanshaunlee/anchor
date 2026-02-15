# Structured synthetic heterograph generator for HGT training.
# Fraud labels from structural generative process (cross-session, motifs, burstiness, device sharing, community).

from ml.synthetic.structured_generator import (
    generate_structured_heterograph,
    StructuredGeneratorConfig,
)

__all__ = ["generate_structured_heterograph", "StructuredGeneratorConfig"]
