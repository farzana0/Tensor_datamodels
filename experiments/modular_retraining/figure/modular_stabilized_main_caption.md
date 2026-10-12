# Main figure: additive-initialized TT and historical comparators

Attribution estimation from limited coalition observations in the modular
relation-table retraining game. TNDM-TT (additive init) is a newly fitted variant
authorized after recovery of the historical results. Linear, quadratic, cubic
OMP, RBF, and TNDM-CP curves are unchanged historical fits; this figure does not
show the original TT initialization, which is retained in the companion figure.
No learner was retrained for the new experiment: only TT surrogates were fitted
to the same measured coalition labels and original matched observation designs.

The new TT initialization starts from a validation-selected additive ridge
surface, applies rank-truncated TT-SVD, and uses the frozen two-restart protocol.
The original ALS optimization, rank/ridge candidates, and validation rule are
otherwise retained. Additive selection introduces additional computation and
validation adaptation but no additional coalition observations. See the paper for complete details and limitations.

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

The convenience main-figure copy selects the **log** y-axis version.

An ordinary base-10 logarithmic y-axis version is supplied because every saved
arithmetic mean-minus-SD bound in this six-method figure is strictly positive.
Bands still represent the original arithmetic mean plus/minus sample SD: the
endpoints are transformed only for display, not recomputed in log space. No
positive floor, clipping, or removal of low/error endpoints is used. The full
linear and signed symmetric-log versions remain available alongside this version.
