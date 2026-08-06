# Corpus retired when the decode bounds were widened (PHASE-8)

Corpus entries are **raw byte strings**, and they only mean something under the decode layout
that produced them. Widening `DECODE_BOUNDS` (`max_dim` 8 → 64) and making the element budget
depend on input length changed what those bytes decode to, so every saved input now describes
a different case than the coverage that earned it a place here.

Kept rather than deleted: the bytes are still valid inputs, and if the old bounds are ever
restored the corpus becomes meaningful again. It is *not* a starting point for a campaign
under the new bounds.
