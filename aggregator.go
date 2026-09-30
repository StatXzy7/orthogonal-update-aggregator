// Package aggregator implements a CPU-only low-rank client update merger.
//
// Clients in a round share an n×r basis B. Each client sends only an m×r
// output factor X_i and a positive weight w_i. The server forms the true
// merged update
//
//	Delta = (sum_i w_i X_i / sum_i w_i) Bᵀ
//
// before computing a fresh SVD factorization. Output and input factors are
// never averaged separately. The SVD is written Delta = Out In, where In is
// k×n with orthonormal rows; Inᵀ is the shared basis for the next round.
package aggregator

import (
	"errors"
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

const (
	// MaxDimension bounds input dimension, output dimension, and rank.
	MaxDimension = 12

	equivalenceTolerance = 1e-9
	orthoTolerance       = 1e-9
)

var (
	ErrNoClients         = errors.New("aggregator: no client submissions")
	ErrDuplicateClient   = errors.New("aggregator: duplicate client submission")
	ErrStaleRound        = errors.New("aggregator: stale round")
	ErrBasisMismatch     = errors.New("aggregator: basis version mismatch")
	ErrInvalidWeight     = errors.New("aggregator: weight must be a positive finite number")
	ErrDimensionConflict = errors.New("aggregator: dimension conflict")
	ErrDimensionTooLarge = errors.New("aggregator: dimension exceeds 12")
	ErrNonFinite         = errors.New("aggregator: value is NaN or infinite")
	ErrRankOutOfRange    = errors.New("aggregator: target rank out of range")
	ErrNotOrthonormal    = errors.New("aggregator: matrix is not orthonormal")
	ErrEmptyClientID     = errors.New("aggregator: client id must not be empty")
	ErrNumericalFailure  = errors.New("aggregator: numerical factorization failed")
)

// Submission is one client's output-side factor for the current round.
type Submission struct {
	ClientID     string
	Round        uint64
	BasisVersion int
	Out          *mat.Dense
	PosWeight    float64
}

// MergedDelta is the complete merged update and its fresh factorization.
// In has shape TargetRank×InputDim and orthonormal rows. Basis is Inᵀ.
type MergedDelta struct {
	Round            uint64
	Delta            *mat.Dense
	Out              *mat.Dense
	In               *mat.Dense
	Basis            *mat.Dense
	SingularValues   []float64
	TargetRank       int
	PreviousRank     int
	RankReduced      bool
	Truncated        bool
	ExactEquivalence bool
	TruncationError  float64
	ProductError     float64
	TotalWeight      float64
	NumClients       int
}

// Merger stores the current round, common basis, and pending submissions.
type Merger struct {
	round        uint64
	basisVersion int
	inputDim     int
	rank         int
	outputDim    int
	basis        *mat.Dense
	pending      []Submission
	seen         map[string]struct{}
	lastMerged   *MergedDelta
}

// NewMerger creates a merger. A nil basis uses the first rank identity rows.
func NewMerger(inputDim, rank int, basis *mat.Dense) (*Merger, error) {
	if inputDim < 1 {
		return nil, fmt.Errorf("%w: input dimension %d must be positive", ErrDimensionConflict, inputDim)
	}
	if inputDim > MaxDimension {
		return nil, fmt.Errorf("%w: input dimension %d", ErrDimensionTooLarge, inputDim)
	}
	if rank < 1 || rank > inputDim {
		return nil, fmt.Errorf("%w: rank %d must be in [1, %d]", ErrRankOutOfRange, rank, inputDim)
	}

	b := mat.NewDense(inputDim, rank, nil)
	if basis == nil {
		for i := 0; i < rank; i++ {
			b.Set(i, i, 1)
		}
	} else {
		rows, cols := basis.Dims()
		if rows != inputDim || cols != rank {
			return nil, fmt.Errorf("%w: basis is %dx%d, want %dx%d", ErrDimensionConflict, rows, cols, inputDim, rank)
		}
		if !isFiniteDense(basis) {
			return nil, fmt.Errorf("%w: basis", ErrNonFinite)
		}
		if !orthonormalColumns(basis, orthoTolerance) {
			return nil, ErrNotOrthonormal
		}
		b.Copy(basis)
	}

	return &Merger{
		round:        1,
		basisVersion: 1,
		inputDim:     inputDim,
		rank:         rank,
		basis:        b,
		seen:         make(map[string]struct{}),
	}, nil
}

func (m *Merger) Round() uint64     { return m.round }
func (m *Merger) BasisVersion() int { return m.basisVersion }
func (m *Merger) Rank() int         { return m.rank }
func (m *Merger) InputDim() int     { return m.inputDim }
func (m *Merger) OutputDim() int    { return m.outputDim }
func (m *Merger) Basis() *mat.Dense { return denseCopyOf(m.basis) }
func (m *Merger) PendingCount() int { return len(m.pending) }

func (m *Merger) Pending() []Submission {
	submissions := make([]Submission, len(m.pending))
	for i, submission := range m.pending {
		submissions[i] = submission
		submissions[i].Out = denseCopyOf(submission.Out)
	}
	return submissions
}

func (m *Merger) LastMerged() (*MergedDelta, bool) {
	if m.lastMerged == nil {
		return nil, false
	}
	return copyMerged(m.lastMerged), true
}

// Submit accepts a contribution only for the current round and basis.
func (m *Merger) Submit(submission Submission) error {
	if submission.ClientID == "" {
		return ErrEmptyClientID
	}
	if submission.Round != m.round {
		return fmt.Errorf("%w: current round %d, submission round %d", ErrStaleRound, m.round, submission.Round)
	}
	if submission.BasisVersion != m.basisVersion {
		return fmt.Errorf("%w: current basis %d, submission basis %d", ErrBasisMismatch, m.basisVersion, submission.BasisVersion)
	}
	if _, exists := m.seen[submission.ClientID]; exists {
		return fmt.Errorf("%w: client %q", ErrDuplicateClient, submission.ClientID)
	}
	if !positiveFinite(submission.PosWeight) {
		return fmt.Errorf("%w: got %v", ErrInvalidWeight, submission.PosWeight)
	}
	if submission.Out == nil {
		return fmt.Errorf("%w: missing output factor", ErrDimensionConflict)
	}

	rows, cols := submission.Out.Dims()
	if rows < 1 || cols < 1 {
		return fmt.Errorf("%w: output factor is %dx%d", ErrDimensionConflict, rows, cols)
	}
	if rows > MaxDimension {
		return fmt.Errorf("%w: output dimension %d", ErrDimensionTooLarge, rows)
	}
	if cols > MaxDimension {
		return fmt.Errorf("%w: output factor rank %d", ErrDimensionTooLarge, cols)
	}
	if cols != m.rank {
		return fmt.Errorf("%w: output factor has %d columns, current rank is %d", ErrDimensionConflict, cols, m.rank)
	}
	if m.outputDim != 0 && rows != m.outputDim {
		return fmt.Errorf("%w: output dimension %d does not match %d", ErrDimensionConflict, rows, m.outputDim)
	}
	if !isFiniteDense(submission.Out) {
		return fmt.Errorf("%w: output factor for client %q", ErrNonFinite, submission.ClientID)
	}

	copied := submission
	copied.Out = denseCopyOf(submission.Out)
	m.seen[submission.ClientID] = struct{}{}
	m.pending = append(m.pending, copied)
	if m.outputDim == 0 {
		m.outputDim = rows
	}
	return nil
}

// MergeDefault keeps the current target rank and requires product equivalence.
func (m *Merger) MergeDefault() (*MergedDelta, error) {
	return m.Merge(m.rank)
}

// Merge combines every pending submission, then refactorizes the full delta.
func (m *Merger) Merge(targetRank int) (*MergedDelta, error) {
	if len(m.pending) == 0 {
		return nil, ErrNoClients
	}
	if targetRank < 1 || targetRank > m.rank {
		return nil, fmt.Errorf("%w: %d must be in [1, %d]", ErrRankOutOfRange, targetRank, m.rank)
	}

	previousRank := m.rank
	totalWeight, err := totalPositiveWeight(m.pending)
	if err != nil {
		return nil, err
	}

	weighted := mat.NewDense(m.outputDim, m.rank, nil)
	scaled := mat.NewDense(m.outputDim, m.rank, nil)
	for _, submission := range m.pending {
		scaled.Scale(submission.PosWeight/totalWeight, submission.Out)
		weighted.Add(weighted, scaled)
	}
	if !isFiniteDense(weighted) {
		return nil, fmt.Errorf("%w: weighted output factor", ErrNonFinite)
	}

	delta := mat.NewDense(m.outputDim, m.inputDim, nil)
	delta.Mul(weighted, m.basis.T())
	if !isFiniteDense(delta) {
		return nil, fmt.Errorf("%w: merged delta", ErrNonFinite)
	}

	out, in, singularValues, truncationError, err := factorizeDelta(delta, targetRank)
	if err != nil {
		return nil, errors.Join(ErrNumericalFailure, err)
	}

	product := mat.NewDense(m.outputDim, m.inputDim, nil)
	product.Mul(out, in)
	if !isFiniteDense(product) {
		return nil, fmt.Errorf("%w: reconstructed product", ErrNumericalFailure)
	}
	productError := frobeniusDiff(delta, product)
	if !positiveOrZeroFinite(productError) {
		return nil, fmt.Errorf("%w: non-finite product residual", ErrNumericalFailure)
	}

	deltaNorm := frobeniusNorm(delta)
	tolerance := equivalenceTolerance * math.Max(1, deltaNorm)
	if math.Abs(productError-truncationError) > tolerance {
		return nil, fmt.Errorf("%w: residual %g does not match truncation error %g", ErrNumericalFailure, productError, truncationError)
	}
	if !orthonormalRows(in, orthoTolerance) {
		return nil, fmt.Errorf("%w: input factor rows", ErrNumericalFailure)
	}

	rankReduced := targetRank < previousRank
	exact := !rankReduced && productError <= tolerance
	if !rankReduced && !exact {
		return nil, fmt.Errorf("%w: default-rank residual %g exceeds tolerance %g", ErrNumericalFailure, productError, tolerance)
	}

	newBasis := transposeCopy(in)
	merged := &MergedDelta{
		Round:            m.round,
		Delta:            delta,
		Out:              out,
		In:               in,
		Basis:            newBasis,
		SingularValues:   singularValues,
		TargetRank:       targetRank,
		PreviousRank:     previousRank,
		RankReduced:      rankReduced,
		Truncated:        rankReduced,
		ExactEquivalence: exact,
		TruncationError:  truncationError,
		ProductError:     productError,
		TotalWeight:      totalWeight,
		NumClients:       len(m.pending),
	}

	m.basis = denseCopyOf(newBasis)
	m.rank = targetRank
	m.round++
	m.basisVersion++
	m.pending = nil
	m.seen = make(map[string]struct{})
	m.lastMerged = copyMerged(merged)
	return merged, nil
}

func totalPositiveWeight(submissions []Submission) (float64, error) {
	var total float64
	for _, submission := range submissions {
		if !positiveFinite(submission.PosWeight) {
			return 0, fmt.Errorf("%w: got %v", ErrInvalidWeight, submission.PosWeight)
		}
		if submission.PosWeight > math.MaxFloat64-total {
			return 0, fmt.Errorf("%w: total positive-sample weight overflows", ErrNonFinite)
		}
		total += submission.PosWeight
	}
	if !positiveFinite(total) {
		return 0, fmt.Errorf("%w: total weight is %v", ErrNonFinite, total)
	}
	return total, nil
}

func copyMerged(in *MergedDelta) *MergedDelta {
	out := *in
	out.Delta = denseCopyOf(in.Delta)
	out.Out = denseCopyOf(in.Out)
	out.In = denseCopyOf(in.In)
	out.Basis = denseCopyOf(in.Basis)
	out.SingularValues = append([]float64(nil), in.SingularValues...)
	return &out
}

func positiveFinite(v float64) bool {
	return !math.IsNaN(v) && !math.IsInf(v, 0) && v > 0
}

func positiveOrZeroFinite(v float64) bool {
	return !math.IsNaN(v) && !math.IsInf(v, 0) && v >= 0
}
