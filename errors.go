// Sentinel errors describe rejected submissions, failed merges, and invalid snapshots.
package aggregator

import "errors"

var (
	ErrInvalidDimension     = errors.New("invalid matrix dimension")
	ErrDimensionLimit       = errors.New("matrix dimension exceeds limit")
	ErrShapeMismatch        = errors.New("matrix shape mismatch")
	ErrNonFiniteValue       = errors.New("non-finite value")
	ErrInvalidWeight        = errors.New("invalid client weight")
	ErrInvalidClient        = errors.New("invalid client id")
	ErrDuplicateClient      = errors.New("duplicate client in current round")
	ErrRoundMismatch        = errors.New("round does not match active round")
	ErrBasisVersionMismatch = errors.New("basis version does not match active basis")
	ErrNoSubmissions        = errors.New("no client submissions for round")
	ErrInvalidTargetRank    = errors.New("invalid target rank")
	ErrInvalidBasis         = errors.New("input basis is not row-orthonormal")
	ErrInvalidTolerance     = errors.New("invalid tolerance")
	ErrDecompositionFailed  = errors.New("increment decomposition failed")
	ErrEquivalenceFailed    = errors.New("reparameterized product is not equivalent to merged increment")
	ErrNoMergedIncrement    = errors.New("no increment has been merged yet")
	ErrInvalidState         = errors.New("invalid persisted aggregator state")
)
