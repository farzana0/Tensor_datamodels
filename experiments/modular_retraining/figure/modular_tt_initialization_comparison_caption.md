# Companion figure: original versus additive-informed TT initialization

Original historical TNDM-TT fits and newly fitted TNDM-TT (additive init) on the
same three learner seeds, five budgets, and twenty observation repetitions per
seed/budget. The original means and SDs, including low-budget instabilities, are
retained intact. This comparison is descriptive evidence for one prespecified
initialization variant on the same saved modular game, not a new task replication.

Each point is the equal-weight mean of three learner-seed means (seeds
31, 43, and 59), each computed across 20 matched coalition-sampling repetitions.
Light translucent bands are plus/minus one sample SD across the three seed means;
they are not confidence intervals or the SD of 60 pooled repetitions. The primary
query is triple_ABC, with Shapley, Banzhaf, and pairwise Shapley interaction-index
(SII) relative L2 errors evaluated against each seed's complete d=8 measured cube.
The new fits jointly predict all four query outputs, as in the original protocol.

The x-axis uses base-2 logarithmic spacing and labels all five budgets. Both full
linear and signed symmetric-log y-axis versions are supplied. The latter is
linear on [-1, 1]. Every saved mean-minus-SD and mean-plus-SD bound is preserved;
negative lower bounds describe symmetric SD bands, not negative individual errors.
No unfavorable original-TT runs are dropped from the initialization comparison.

DejaVu Sans 9 pt, 8.5 by 2.9 inches, and light bands (alpha 0.12) follow the recovered
fallback style. Exact matching to the unavailable v64/HAR-SD source is unverified.
No overall title, subtitle, caption, or footer is embedded in the figure.

The convenience initialization-comparison copy uses a signed symmetric-log
y-axis to retain the original TT's negative lower SD endpoints. No ordinary
logarithmic y-axis comparison is generated.
