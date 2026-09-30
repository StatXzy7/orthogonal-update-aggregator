package aggregator

import (
	"errors"
	"math"
	"sync"

	"gonum.org/v1/gonum/mat"
)

// MaxDimension is the maximum permitted output or input dimension.
const MaxDimension = 12

// Matrix is the accepted dense matrix interface.
type Matrix = mat.Matrix

// Submission is one client output-side update for the active round and basis.
type Submission struct {
	Round        int
	BasisVersion int
	ClientID     string
	OutputFactor Matrix
	Weight       float64
}

// MergeResult reports the merged increment, accepted approximation, and diagnostics.
type MergeResult struct {
	CompletedRound          int
	NextRound               int
	BasisVersionBefore      int
	BasisVersionAfter       int
	PreviousRank            int
	RequestedRank           int
	NewRank                 int
	Truncated               bool
	TruncationError         float64
	RelativeTruncationError float64
	ProductResidual         float64
	OrthogonalityError      float64
	Equivalent              bool
	SingularValues          []float64
	FullDelta               *mat.Dense
	AcceptedDelta           *mat.Dense
	InputBasis              *mat.Dense
	OutputFactor            *mat.Dense
}

type submitted struct {
	Round        int
	BasisVersion int
	ClientID     string
	Weight       float64
	OutputFactor *mat.Dense
}

// Aggregator tracks rounds, the shared input basis, factors, and pending submissions.
type Aggregator struct {
	mu           sync.RWMutex
	round        int
	basisVersion int
	outputDim    int
	inputDim     int
	rank         int
	tolerance    float64
	inputBasis   *mat.Dense
	outputFactor *mat.Dense
	pending      map[string]*submitted
	lastMerge    *MergeResult
}

type Option func(*config)

type config struct {
	tolerance float64
}

// WithTolerance sets the product and orthogonality numeric tolerance.
func WithTolerance(tolerance float64) Option {
	return func(c *config) {
		c.tolerance = tolerance
	}
}

// New creates an aggregator with a row-orthonormal rank-by-inputDim initial basis.
func New(outputDim, inputDim, rank int, inputBasis Matrix, options ...Option) (*Aggregator, error) {
	cfg := config{tolerance: 1e-10}
	for _, option := range options {
		option(&cfg)
	}
	if outputDim <= 0 || outputDim > MaxDimension || inputDim <= 0 || inputDim > MaxDimension || rank <= 0 || rank > MaxDimension || rank > inputDim || rank > outputDim {
		return nil, ErrInvalidDimension
	}
	if !isFinite(cfg.tolerance) || cfg.tolerance <= 0 {
		return nil, ErrInvalidTolerance
	}
	basis, err := denseCopy(inputBasis)
	if err != nil {
		return nil, err
	}
	if r, c := basis.Dims(); r != rank || c != inputDim {
		return nil, errors.Join(ErrShapeMismatch, ErrInvalidBasis)
	}
	if err := validateFinite(basis); err != nil {
		return nil, err
	}
	if orthogonalityError(basis) > cfg.tolerance {
		return nil, ErrInvalidBasis
	}

	return &Aggregator{
		round:        1,
		basisVersion: 1,
		outputDim:    outputDim,
		inputDim:     inputDim,
		rank:         rank,
		tolerance:    cfg.tolerance,
		inputBasis:   basis,
		outputFactor: mat.NewDense(outputDim, rank, nil),
		pending:      make(map[string]*submitted),
	}, nil
}

// Round returns the active round awaiting submissions.
func (a *Aggregator) Round() int {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.round
}

// BasisVersion returns the version of the active shared input basis.
func (a *Aggregator) BasisVersion() int {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.basisVersion
}

// Rank returns the current target factor rank.
func (a *Aggregator) Rank() int {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.rank
}

// Dims returns output dimension, input dimension, and current rank.
func (a *Aggregator) Dims() (outputDim, inputDim, rank int) {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.outputDim, a.inputDim, a.rank
}

// Tolerance returns the configured numeric tolerance.
func (a *Aggregator) Tolerance() float64 {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.tolerance
}

// PendingClientCount returns the number of validated submissions awaiting merge.
func (a *Aggregator) PendingClientCount() int {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return len(a.pending)
}

// PendingClients returns sorted IDs of clients with pending submissions.
func (a *Aggregator) PendingClients() []string {
	a.mu.RLock()
	defer a.mu.RUnlock()
	clients := make([]string, 0, len(a.pending))
	for client := range a.pending {
		clients = append(clients, client)
	}
	sortStrings(clients)
	return clients
}

// InputBasis returns a defensive copy of the shared row-orthonormal basis.
func (a *Aggregator) InputBasis() *mat.Dense {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return denseClone(a.inputBasis)
}

// OutputFactor returns a defensive copy of the current output factor.
func (a *Aggregator) OutputFactor() *mat.Dense {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return denseClone(a.outputFactor)
}

// CurrentIncrement returns the current factor product C*B.
func (a *Aggregator) CurrentIncrement() (*mat.Dense, error) {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return multiply(a.outputFactor, a.inputBasis)
}

// MergedIncrement returns the complete weighted increment from the last successful round.
func (a *Aggregator) MergedIncrement() (*mat.Dense, error) {
	a.mu.RLock()
	defer a.mu.RUnlock()
	if a.lastMerge == nil || a.lastMerge.FullDelta == nil {
		return nil, ErrNoMergedIncrement
	}
	return denseClone(a.lastMerge.FullDelta), nil
}

// LastMerge returns diagnostics and factors from the last successful merge.
func (a *Aggregator) LastMerge() (*MergeResult, error) {
	a.mu.RLock()
	defer a.mu.RUnlock()
	if a.lastMerge == nil {
		return nil, ErrNoMergedIncrement
	}
	return cloneMergeResult(a.lastMerge), nil
}

// Submit validates and stores one output factor for the active round and basis.
func (a *Aggregator) Submit(submission Submission) error {
	if submission.Round <= 0 || submission.BasisVersion <= 0 {
		return ErrRoundMismatch
	}
	if submission.ClientID == "" {
		return ErrInvalidClient
	}
	if !isFinite(submission.Weight) || submission.Weight <= 0 {
		return ErrInvalidWeight
	}
	factor, err := denseCopy(submission.OutputFactor)
	if err != nil {
		return err
	}

	a.mu.Lock()
	defer a.mu.Unlock()

	if submission.Round != a.round {
		return ErrRoundMismatch
	}
	if submission.BasisVersion != a.basisVersion {
		return ErrBasisVersionMismatch
	}
	if _, exists := a.pending[submission.ClientID]; exists {
		return ErrDuplicateClient
	}
	if r, c := factor.Dims(); r != a.outputDim || c != a.rank {
		return ErrShapeMismatch
	}
	if err := validateFinite(factor); err != nil {
		return err
	}

	a.pending[submission.ClientID] = &submitted{
		Round:        submission.Round,
		BasisVersion: submission.BasisVersion,
		ClientID:     submission.ClientID,
		Weight:       submission.Weight,
		OutputFactor: factor,
	}
	return nil
}

// CommitRound merges all pending submissions, then advances only on full numeric success.
func (a *Aggregator) CommitRound(targetRank int) (result *MergeResult, err error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.commitRoundLocked(targetRank)
}

func (a *Aggregator) commitRoundLocked(targetRank int) (result *MergeResult, err error) {
	if err := a.validatePendingLocked(targetRank); err != nil {
		return nil, err
	}
	candidate := a.rank
	if targetRank != 0 {
		candidate = targetRank
	}

	oldBasis := a.inputBasis
	oldFactor := a.outputFactor
	oldRank := a.rank
	oldPending := clonePendingLocked(a.pending)
	oldLast := a.lastMerge
	committed := false
	defer func() {
		if recovered := recover(); recovered != nil && !committed {
			a.inputBasis = oldBasis
			a.outputFactor = oldFactor
			a.rank = oldRank
			a.pending = oldPending
			a.lastMerge = oldLast
			result = nil
			err = ErrDecompositionFailed
		}
	}()

	result, newBasis, newFactor, err := mergePendingLocked(a.pending, a.inputBasis, a.outputDim, a.inputDim, a.round, a.basisVersion, a.rank, targetRank, a.tolerance)
	if err != nil {
		a.inputBasis = oldBasis
		a.outputFactor = oldFactor
		a.rank = oldRank
		a.pending = oldPending
		a.lastMerge = oldLast
		return nil, err
	}

	a.inputBasis = newBasis
	a.outputFactor = newFactor
	a.rank = candidate
	a.round++
	a.basisVersion++
	a.pending = make(map[string]*submitted)
	a.lastMerge = result
	committed = true
	return cloneMergeResult(result), nil
}

func (a *Aggregator) validatePendingLocked(targetRank int) error {
	if len(a.pending) == 0 {
		return ErrNoSubmissions
	}
	requested := targetRank
	if requested == 0 {
		requested = a.rank
	}
	if requested <= 0 || requested > a.rank {
		return ErrInvalidTargetRank
	}
	for client, item := range a.pending {
		if item.Round != a.round {
			return ErrRoundMismatch
		}
		if item.BasisVersion != a.basisVersion {
			return ErrBasisVersionMismatch
		}
		if item.ClientID != client || client == "" {
			return ErrInvalidClient
		}
		if !isFinite(item.Weight) || item.Weight <= 0 {
			return ErrInvalidWeight
		}
		if item.OutputFactor == nil {
			return ErrShapeMismatch
		}
		if r, c := item.OutputFactor.Dims(); r != a.outputDim || c != a.rank {
			return ErrShapeMismatch
		}
		if err := validateFinite(item.OutputFactor); err != nil {
			return err
		}
	}
	return nil
}

func mergePendingLocked(pending map[string]*submitted, basis *mat.Dense, outputDim, inputDim, round, basisVersion, oldRank, targetRank int, tolerance float64) (*MergeResult, *mat.Dense, *mat.Dense, error) {
	if targetRank == 0 {
		targetRank = oldRank
	}
	basisRows, basisCols := basis.Dims()
	if basisRows != oldRank || basisCols != inputDim {
		return nil, nil, nil, ErrShapeMismatch
	}

	clients := make([]string, 0, len(pending))
	totalWeight := 0.0
	for client, item := range pending {
		clients = append(clients, client)
		totalWeight += item.Weight
	}
	sortStrings(clients)
	if !isFinite(totalWeight) {
		return nil, nil, nil, ErrNonFiniteValue
	}
	if totalWeight <= 0 {
		return nil, nil, nil, ErrInvalidWeight
	}

	mergedOutput := mat.NewDense(outputDim, oldRank, nil)
	for _, client := range clients {
		item := pending[client]
		var scaled mat.Dense
		scaled.Scale(item.Weight/totalWeight, item.OutputFactor)
		mergedOutput.Add(mergedOutput, &scaled)
	}
	if err := validateFinite(mergedOutput); err != nil {
		return nil, nil, nil, err
	}

	fullDelta, err := multiply(mergedOutput, basis)
	if err != nil {
		return nil, nil, nil, err
	}
	newBasis, newFactor, acceptedDelta, singularValues, residual, orthError, truncationError, err := refactorIncrement(fullDelta, outputDim, inputDim, oldRank, targetRank, tolerance)
	if err != nil {
		return nil, nil, nil, err
	}

	truncated := targetRank < oldRank
	deltaNorm := frobeniusNorm(fullDelta)
	result := &MergeResult{
		CompletedRound:          round,
		NextRound:               round + 1,
		BasisVersionBefore:      basisVersion,
		BasisVersionAfter:       basisVersion + 1,
		PreviousRank:            oldRank,
		RequestedRank:           targetRank,
		NewRank:                 targetRank,
		Truncated:               truncated,
		TruncationError:         truncationError,
		RelativeTruncationError: relativeError(truncationError, deltaNorm),
		ProductResidual:         residual,
		OrthogonalityError:      orthError,
		Equivalent:              !truncated && residual <= tolerance*math.Max(1, deltaNorm) && orthError <= tolerance,
		SingularValues:          append([]float64(nil), singularValues...),
		FullDelta:               denseClone(fullDelta),
		AcceptedDelta:           denseClone(acceptedDelta),
		InputBasis:              denseClone(newBasis),
		OutputFactor:            denseClone(newFactor),
	}
	return result, newBasis, newFactor, nil
}

func relativeError(errorValue, scale float64) float64 {
	if scale == 0 || !isFinite(scale) {
		return 0
	}
	return errorValue / scale
}
