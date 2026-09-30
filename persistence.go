package aggregator

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"gonum.org/v1/gonum/mat"
)

const stateFileVersion = 1

type matrixState struct {
	Rows int       `json:"rows"`
	Cols int       `json:"cols"`
	Data []float64 `json:"data"`
}

type submissionState struct {
	ClientID     string      `json:"client_id"`
	Round        uint64      `json:"round"`
	BasisVersion int         `json:"basis_version"`
	PosWeight    float64     `json:"pos_weight"`
	Out          matrixState `json:"out"`
}

type mergedState struct {
	Round            uint64      `json:"round"`
	Delta            matrixState `json:"delta"`
	Out              matrixState `json:"out"`
	In               matrixState `json:"in"`
	Basis            matrixState `json:"basis"`
	SingularValues   []float64   `json:"singular_values"`
	TargetRank       int         `json:"target_rank"`
	PreviousRank     int         `json:"previous_rank"`
	RankReduced      bool        `json:"rank_reduced"`
	Truncated        bool        `json:"truncated"`
	ExactEquivalence bool        `json:"exact_equivalence"`
	TruncationError  float64     `json:"truncation_error"`
	ProductError     float64     `json:"product_error"`
	TotalWeight      float64     `json:"total_weight"`
	NumClients       int         `json:"num_clients"`
}

type mergerState struct {
	Version      int               `json:"version"`
	Round        uint64            `json:"round"`
	BasisVersion int               `json:"basis_version"`
	InputDim     int               `json:"input_dim"`
	Rank         int               `json:"rank"`
	OutputDim    int               `json:"output_dim"`
	Basis        matrixState       `json:"basis"`
	Pending      []submissionState `json:"pending"`
	LastMerged   *mergedState      `json:"last_merged,omitempty"`
}

func matrixStateOf(m *mat.Dense) matrixState {
	rows, cols := m.Dims()
	data := make([]float64, 0, rows*cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			data = append(data, m.At(i, j))
		}
	}
	return matrixState{Rows: rows, Cols: cols, Data: data}
}

func (s matrixState) dense() (*mat.Dense, error) {
	if s.Rows < 1 || s.Rows > MaxDimension || s.Cols < 1 || s.Cols > MaxDimension {
		return nil, fmt.Errorf("%w: stored matrix is %dx%d", ErrDimensionConflict, s.Rows, s.Cols)
	}
	if len(s.Data) != s.Rows*s.Cols {
		return nil, fmt.Errorf("%w: stored matrix has %d values, want %d", ErrDimensionConflict, len(s.Data), s.Rows*s.Cols)
	}
	matrix := mat.NewDense(s.Rows, s.Cols, append([]float64(nil), s.Data...))
	if !isFiniteDense(matrix) {
		return nil, fmt.Errorf("%w: stored matrix", ErrNonFinite)
	}
	return matrix, nil
}

// Save writes a fully restorable snapshot and atomically replaces path.
func (m *Merger) Save(path string) error {
	state := mergerState{
		Version:      stateFileVersion,
		Round:        m.round,
		BasisVersion: m.basisVersion,
		InputDim:     m.inputDim,
		Rank:         m.rank,
		OutputDim:    m.outputDim,
		Basis:        matrixStateOf(m.basis),
	}

	for _, submission := range m.pending {
		state.Pending = append(state.Pending, submissionState{
			ClientID:     submission.ClientID,
			Round:        submission.Round,
			BasisVersion: submission.BasisVersion,
			PosWeight:    submission.PosWeight,
			Out:          matrixStateOf(submission.Out),
		})
	}

	if m.lastMerged != nil {
		merged := m.lastMerged
		state.LastMerged = &mergedState{
			Round:            merged.Round,
			Delta:            matrixStateOf(merged.Delta),
			Out:              matrixStateOf(merged.Out),
			In:               matrixStateOf(merged.In),
			Basis:            matrixStateOf(merged.Basis),
			SingularValues:   append([]float64(nil), merged.SingularValues...),
			TargetRank:       merged.TargetRank,
			PreviousRank:     merged.PreviousRank,
			RankReduced:      merged.RankReduced,
			Truncated:        merged.Truncated,
			ExactEquivalence: merged.ExactEquivalence,
			TruncationError:  merged.TruncationError,
			ProductError:     merged.ProductError,
			TotalWeight:      merged.TotalWeight,
			NumClients:       merged.NumClients,
		}
	}

	buf, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return fmt.Errorf("aggregator: encode state: %w", err)
	}

	dir := filepath.Dir(path)
	file, err := os.CreateTemp(dir, "."+filepath.Base(path)+".tmp-*")
	if err != nil {
		return fmt.Errorf("aggregator: create temporary state: %w", err)
	}
	tmpPath := file.Name()
	removeTemp := true
	defer func() {
		if removeTemp {
			_ = os.Remove(tmpPath)
		}
	}()

	if _, err := file.Write(buf); err != nil {
		_ = file.Close()
		return fmt.Errorf("aggregator: write state: %w", err)
	}
	if err := file.Sync(); err != nil {
		_ = file.Close()
		return fmt.Errorf("aggregator: sync state: %w", err)
	}
	if err := file.Close(); err != nil {
		return fmt.Errorf("aggregator: close state: %w", err)
	}
	if err := os.Chmod(tmpPath, 0o600); err != nil {
		return fmt.Errorf("aggregator: set state permissions: %w", err)
	}
	if err := os.Rename(tmpPath, path); err != nil {
		return fmt.Errorf("aggregator: replace state file: %w", err)
	}
	removeTemp = false
	return nil
}

// Load restores and revalidates a snapshot written by Save.
func Load(path string) (*Merger, error) {
	buf, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("aggregator: read state: %w", err)
	}

	var state mergerState
	if err := json.Unmarshal(buf, &state); err != nil {
		return nil, fmt.Errorf("aggregator: decode state: %w", err)
	}
	if state.Version != stateFileVersion {
		return nil, fmt.Errorf("%w: unsupported state version %d", ErrDimensionConflict, state.Version)
	}
	if state.Round == 0 || state.BasisVersion <= 0 {
		return nil, fmt.Errorf("%w: invalid round or basis version", ErrDimensionConflict)
	}
	if state.Round != uint64(state.BasisVersion) {
		return nil, fmt.Errorf("%w: round and basis version disagree", ErrBasisMismatch)
	}
	if state.OutputDim < 0 || state.OutputDim > MaxDimension {
		return nil, fmt.Errorf("%w: stored output dimension %d", ErrDimensionConflict, state.OutputDim)
	}
	if state.Round > 1 && state.LastMerged == nil {
		return nil, fmt.Errorf("%w: advanced round has no merged delta", ErrStaleRound)
	}
	if state.Round == 1 && state.LastMerged != nil {
		return nil, fmt.Errorf("%w: initial round unexpectedly has merged delta", ErrStaleRound)
	}
	if state.LastMerged != nil && state.OutputDim == 0 {
		return nil, fmt.Errorf("%w: merged delta has no output dimension", ErrDimensionConflict)
	}

	basis, err := state.Basis.dense()
	if err != nil {
		return nil, fmt.Errorf("aggregator: invalid stored basis: %w", err)
	}
	merger, err := NewMerger(state.InputDim, state.Rank, basis)
	if err != nil {
		return nil, fmt.Errorf("aggregator: invalid stored state: %w", err)
	}
	merger.round = state.Round
	merger.basisVersion = state.BasisVersion
	merger.outputDim = state.OutputDim

	for _, stored := range state.Pending {
		out, err := stored.Out.dense()
		if err != nil {
			return nil, fmt.Errorf("aggregator: invalid submission for client %q: %w", stored.ClientID, err)
		}
		if err := merger.Submit(Submission{
			ClientID:     stored.ClientID,
			Round:        stored.Round,
			BasisVersion: stored.BasisVersion,
			Out:          out,
			PosWeight:    stored.PosWeight,
		}); err != nil {
			return nil, fmt.Errorf("aggregator: invalid stored submissions: %w", err)
		}
	}

	if state.LastMerged != nil {
		merged, err := state.LastMerged.merged(state.InputDim, state.Rank, state.OutputDim, state.Round, state.BasisVersion, basis)
		if err != nil {
			return nil, err
		}
		merger.lastMerged = merged
	}

	return merger, nil
}

func (s *mergedState) merged(inputDim, currentRank, outputDim int, round uint64, basisVersion int, basis *mat.Dense) (*MergedDelta, error) {
	delta, err := s.Delta.dense()
	if err != nil {
		return nil, fmt.Errorf("aggregator: invalid stored delta: %w", err)
	}
	out, err := s.Out.dense()
	if err != nil {
		return nil, fmt.Errorf("aggregator: invalid stored output factor: %w", err)
	}
	in, err := s.In.dense()
	if err != nil {
		return nil, fmt.Errorf("aggregator: invalid stored input factor: %w", err)
	}
	storedBasis, err := s.Basis.dense()
	if err != nil {
		return nil, fmt.Errorf("aggregator: invalid stored merged basis: %w", err)
	}

	deltaRows, deltaCols := delta.Dims()
	outRows, outCols := out.Dims()
	inRows, inCols := in.Dims()
	basisRows, basisCols := basis.Dims()
	if deltaCols != inputDim || deltaRows != outputDim ||
		outRows != outputDim || outCols != currentRank ||
		inRows != currentRank || inCols != inputDim ||
		basisRows != inputDim || basisCols != currentRank {
		return nil, fmt.Errorf("%w: stored factor dimensions are inconsistent", ErrDimensionConflict)
	}
	storedBasisRows, storedBasisCols := storedBasis.Dims()
	if storedBasisRows != basisRows || storedBasisCols != basisCols || frobeniusDiff(storedBasis, basis) > equivalenceTolerance {
		return nil, fmt.Errorf("%w: stored current basis does not follow last merged factor", ErrDimensionConflict)
	}

	expectedTranspose := transposeCopy(in)
	if frobeniusDiff(expectedTranspose, storedBasis) > equivalenceTolerance {
		return nil, fmt.Errorf("%w: stored basis is not input-factor transpose", ErrNotOrthonormal)
	}
	if !orthonormalRows(in, orthoTolerance) {
		return nil, ErrNotOrthonormal
	}

	if s.Round == 0 || s.Round+1 != round || basisVersion < 2 {
		return nil, fmt.Errorf("%w: stored merged round does not precede current round", ErrStaleRound)
	}
	if s.TargetRank != currentRank || currentRank < 1 || currentRank > s.PreviousRank || s.PreviousRank > inputDim {
		return nil, fmt.Errorf("%w: stored ranks are inconsistent", ErrRankOutOfRange)
	}
	if len(s.SingularValues) != currentRank {
		return nil, fmt.Errorf("%w: stored singular value count", ErrRankOutOfRange)
	}
	for _, sigma := range s.SingularValues {
		if !positiveOrZeroFinite(sigma) {
			return nil, fmt.Errorf("%w: stored singular value", ErrNonFinite)
		}
	}
	for i := 1; i < len(s.SingularValues); i++ {
		if s.SingularValues[i] > s.SingularValues[i-1] {
			return nil, fmt.Errorf("%w: stored singular values are not sorted", ErrNumericalFailure)
		}
	}
	if s.NumClients <= 0 || !positiveFinite(s.TotalWeight) {
		return nil, fmt.Errorf("%w: stored client count or weight", ErrInvalidWeight)
	}
	if !positiveOrZeroFinite(s.TruncationError) || !positiveOrZeroFinite(s.ProductError) {
		return nil, fmt.Errorf("%w: stored error is non-finite", ErrNonFinite)
	}

	tolerance := equivalenceTolerance * math.Max(1, frobeniusNorm(delta))
	var recomputedSVD mat.SVD
	if ok := recomputedSVD.Factorize(delta, mat.SVDFull); !ok {
		return nil, fmt.Errorf("%w: cannot verify stored singular spectrum", ErrNumericalFailure)
	}
	trueValues := recomputedSVD.Values(nil)
	for i := range s.SingularValues {
		want := 0.0
		if i < len(trueValues) {
			want = trueValues[i]
		}
		if math.Abs(s.SingularValues[i]-want) > tolerance {
			return nil, fmt.Errorf("%w: stored singular value %d is inconsistent", ErrNumericalFailure, i)
		}
	}
	var tailSquared float64
	for i := currentRank; i < len(trueValues); i++ {
		tailSquared += trueValues[i] * trueValues[i]
	}
	trueTail := math.Sqrt(tailSquared)
	if math.Abs(trueTail-s.TruncationError) > tolerance {
		return nil, fmt.Errorf("%w: stored truncation error is inconsistent", ErrNumericalFailure)
	}

	product := mat.NewDense(outRows, deltaCols, nil)
	product.Mul(out, in)
	productError := frobeniusDiff(delta, product)
	if math.Abs(productError-s.ProductError) > tolerance || math.Abs(productError-s.TruncationError) > tolerance {
		return nil, fmt.Errorf("%w: stored product does not match reported residual", ErrNumericalFailure)
	}

	rankReduced := currentRank < s.PreviousRank
	if rankReduced != s.RankReduced || rankReduced != s.Truncated {
		return nil, fmt.Errorf("%w: stored truncation flags are inconsistent", ErrNumericalFailure)
	}
	exact := !rankReduced && productError <= tolerance
	if exact != s.ExactEquivalence {
		return nil, fmt.Errorf("%w: stored equivalence flag is inconsistent", ErrNumericalFailure)
	}

	return &MergedDelta{
		Round:            s.Round,
		Delta:            delta,
		Out:              out,
		In:               in,
		Basis:            storedBasis,
		SingularValues:   append([]float64(nil), s.SingularValues...),
		TargetRank:       s.TargetRank,
		PreviousRank:     s.PreviousRank,
		RankReduced:      s.RankReduced,
		Truncated:        s.Truncated,
		ExactEquivalence: s.ExactEquivalence,
		TruncationError:  s.TruncationError,
		ProductError:     s.ProductError,
		TotalWeight:      s.TotalWeight,
		NumClients:       s.NumClients,
	}, nil
}
