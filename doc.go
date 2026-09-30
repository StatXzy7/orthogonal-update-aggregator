// Package aggregator implements a CPU-only low-rank client increment merger.
//
// Every client in a round uses the server's shared row-orthonormal input basis
// B and submits only an output-side factor U and a positive sample weight. The
// server forms the complete increment as the weighted sum of U times B before
// recomputing an SVD-based output factor and row-orthonormal input basis. It
// does not average the two factor sides independently.
//
// CommitRound succeeds only after every pending submission validates and the
// recomputed product satisfies the configured numeric tolerance. A failed
// decomposition leaves the round, basis version, factors, and pending
// submissions unchanged. A lower target rank is treated as truncation:
// MergeResult reports the omitted Frobenius norm and never reports exact
// equivalence.
package aggregator
