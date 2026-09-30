package aggregator

import (
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"

	"gonum.org/v1/gonum/mat"
)

type rawMatrix [][]float64

type rawSubmission struct {
	Round        int       `json:"round"`
	BasisVersion int       `json:"basis_version"`
	ClientID     string    `json:"client_id"`
	Weight       float64   `json:"weight"`
	OutputFactor rawMatrix `json:"output_factor"`
}

type rawMerge struct {
	CompletedRound          int       `json:"completed_round"`
	NextRound               int       `json:"next_round"`
	BasisVersionBefore      int       `json:"basis_version_before"`
	BasisVersionAfter       int       `json:"basis_version_after"`
	PreviousRank            int       `json:"previous_rank"`
	RequestedRank           int       `json:"requested_rank"`
	NewRank                 int       `json:"new_rank"`
	Truncated               bool      `json:"truncated"`
	TruncationError         float64   `json:"truncation_error"`
	RelativeTruncationError float64   `json:"relative_truncation_error"`
	ProductResidual         float64   `json:"product_residual"`
	OrthogonalityError      float64   `json:"orthogonality_error"`
	Equivalent              bool      `json:"equivalent"`
	SingularValues          []float64 `json:"singular_values"`
	FullDelta               rawMatrix `json:"full_delta"`
	AcceptedDelta           rawMatrix `json:"accepted_delta"`
	InputBasis              rawMatrix `json:"input_basis"`
	OutputFactor            rawMatrix `json:"output_factor"`
}

type snapshot struct {
	Version      int                      `json:"version"`
	Round        int                      `json:"round"`
	BasisVersion int                      `json:"basis_version"`
	OutputDim    int                      `json:"output_dim"`
	InputDim     int                      `json:"input_dim"`
	Rank         int                      `json:"rank"`
	Tolerance    float64                  `json:"tolerance"`
	InputBasis   rawMatrix                `json:"input_basis"`
	OutputFactor rawMatrix                `json:"output_factor"`
	Pending      map[string]rawSubmission `json:"pending"`
	LastMerge    *rawMerge                `json:"last_merge,omitempty"`
}

func clonePendingLocked(pending map[string]*submitted) map[string]*submitted {
	result := make(map[string]*submitted, len(pending))
	for key, item := range pending {
		copyItem := *item
		copyItem.OutputFactor = denseClone(item.OutputFactor)
		result[key] = &copyItem
	}
	return result
}

func cloneMergeResult(result *MergeResult) *MergeResult {
	if result == nil {
		return nil
	}
	copyResult := *result
	copyResult.SingularValues = append([]float64(nil), result.SingularValues...)
	copyResult.FullDelta = denseClone(result.FullDelta)
	copyResult.AcceptedDelta = denseClone(result.AcceptedDelta)
	copyResult.InputBasis = denseClone(result.InputBasis)
	copyResult.OutputFactor = denseClone(result.OutputFactor)
	return &copyResult
}

func (a *Aggregator) Save(path string) error {
	a.mu.RLock()
	defer a.mu.RUnlock()

	data, err := json.MarshalIndent(a.snapshotLocked(), "", "  ")
	if err != nil {
		return errors.Join(ErrInvalidState, err)
	}
	dir := filepath.Dir(path)
	temp, err := os.CreateTemp(dir, ".aggregator-state-*.tmp")
	if err != nil {
		return errors.Join(ErrInvalidState, err)
	}
	tempName := temp.Name()
	cleanup := true
	defer func() {
		if cleanup {
			_ = os.Remove(tempName)
		}
	}()
	if _, err := temp.Write(data); err != nil {
		_ = temp.Close()
		return errors.Join(ErrInvalidState, err)
	}
	if err := temp.Chmod(0o600); err != nil {
		_ = temp.Close()
		return errors.Join(ErrInvalidState, err)
	}
	if err := temp.Sync(); err != nil {
		_ = temp.Close()
		return errors.Join(ErrInvalidState, err)
	}
	if err := temp.Close(); err != nil {
		return errors.Join(ErrInvalidState, err)
	}
	if err := os.Rename(tempName, path); err != nil {
		return errors.Join(ErrInvalidState, err)
	}
	cleanup = false
	syncDirectory(dir)
	return nil
}

func Load(path string) (*Aggregator, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, errors.Join(ErrInvalidState, err)
	}
	var snap snapshot
	if err := json.Unmarshal(data, &snap); err != nil {
		return nil, errors.Join(ErrInvalidState, err)
	}
	return aggregatorFromSnapshot(&snap)
}

func (a *Aggregator) snapshotLocked() *snapshot {
	pending := make(map[string]rawSubmission, len(a.pending))
	for key, item := range a.pending {
		pending[key] = rawSubmission{
			Round:        item.Round,
			BasisVersion: item.BasisVersion,
			ClientID:     item.ClientID,
			Weight:       item.Weight,
			OutputFactor: toRawMatrix(item.OutputFactor),
		}
	}
	var last *rawMerge
	if a.lastMerge != nil {
		last = &rawMerge{
			CompletedRound:          a.lastMerge.CompletedRound,
			NextRound:               a.lastMerge.NextRound,
			BasisVersionBefore:      a.lastMerge.BasisVersionBefore,
			BasisVersionAfter:       a.lastMerge.BasisVersionAfter,
			PreviousRank:            a.lastMerge.PreviousRank,
			RequestedRank:           a.lastMerge.RequestedRank,
			NewRank:                 a.lastMerge.NewRank,
			Truncated:               a.lastMerge.Truncated,
			TruncationError:         a.lastMerge.TruncationError,
			RelativeTruncationError: a.lastMerge.RelativeTruncationError,
			ProductResidual:         a.lastMerge.ProductResidual,
			OrthogonalityError:      a.lastMerge.OrthogonalityError,
			Equivalent:              a.lastMerge.Equivalent,
			SingularValues:          append([]float64(nil), a.lastMerge.SingularValues...),
			FullDelta:               toRawMatrix(a.lastMerge.FullDelta),
			AcceptedDelta:           toRawMatrix(a.lastMerge.AcceptedDelta),
			InputBasis:              toRawMatrix(a.lastMerge.InputBasis),
			OutputFactor:            toRawMatrix(a.lastMerge.OutputFactor),
		}
	}
	return &snapshot{
		Version:      1,
		Round:        a.round,
		BasisVersion: a.basisVersion,
		OutputDim:    a.outputDim,
		InputDim:     a.inputDim,
		Rank:         a.rank,
		Tolerance:    a.tolerance,
		InputBasis:   toRawMatrix(a.inputBasis),
		OutputFactor: toRawMatrix(a.outputFactor),
		Pending:      pending,
		LastMerge:    last,
	}
}

func aggregatorFromSnapshot(snap *snapshot) (*Aggregator, error) {
	if snap == nil || snap.Version != 1 {
		return nil, ErrInvalidState
	}
	if snap.OutputDim <= 0 || snap.OutputDim > MaxDimension || snap.InputDim <= 0 || snap.InputDim > MaxDimension || snap.Rank <= 0 || snap.Rank > snap.InputDim || snap.Rank > snap.OutputDim || snap.Round <= 0 || snap.BasisVersion <= 0 || snap.Tolerance <= 0 || !isFinite(snap.Tolerance) {
		return nil, ErrInvalidState
	}
	if snap.BasisVersion != snap.Round {
		return nil, ErrInvalidState
	}
	basis, err := fromRawMatrix(snap.InputBasis, snap.Rank, snap.InputDim)
	if err != nil {
		return nil, errors.Join(ErrInvalidState, err)
	}
	if orthogonalityError(basis) > snap.Tolerance {
		return nil, ErrInvalidState
	}
	factor, err := fromRawMatrix(snap.OutputFactor, snap.OutputDim, snap.Rank)
	if err != nil {
		return nil, errors.Join(ErrInvalidState, err)
	}

	pending := make(map[string]*submitted, len(snap.Pending))
	for key, raw := range snap.Pending {
		if key == "" || raw.ClientID != key {
			return nil, ErrInvalidState
		}
		if raw.Round != snap.Round || raw.BasisVersion != snap.BasisVersion || raw.Weight <= 0 || !isFinite(raw.Weight) {
			return nil, ErrInvalidState
		}
		rawFactor, err := fromRawMatrix(raw.OutputFactor, snap.OutputDim, snap.Rank)
		if err != nil {
			return nil, errors.Join(ErrInvalidState, err)
		}
		pending[key] = &submitted{
			Round:        raw.Round,
			BasisVersion: raw.BasisVersion,
			ClientID:     raw.ClientID,
			Weight:       raw.Weight,
			OutputFactor: rawFactor,
		}
	}

	last, err := mergeFromRaw(snap.LastMerge, snap.OutputDim, snap.InputDim)
	if err != nil {
		return nil, err
	}
	if err := validateSnapshotRelationships(snap, basis, factor, last); err != nil {
		return nil, err
	}
	return &Aggregator{
		round:        snap.Round,
		basisVersion: snap.BasisVersion,
		outputDim:    snap.OutputDim,
		inputDim:     snap.InputDim,
		rank:         snap.Rank,
		tolerance:    snap.Tolerance,
		inputBasis:   basis,
		outputFactor: factor,
		pending:      pending,
		lastMerge:    last,
	}, nil
}

func validateSnapshotRelationships(snap *snapshot, basis, factor *mat.Dense, last *MergeResult) error {
	if last == nil {
		if snap.Round != 1 || snap.BasisVersion != 1 {
			return ErrInvalidState
		}
		return nil
	}
	if snap.Round != last.NextRound || snap.BasisVersion != last.BasisVersionAfter {
		return ErrInvalidState
	}
	if snap.Rank != last.NewRank || !matricesEqual(basis, last.InputBasis, snap.Tolerance) || !matricesEqual(factor, last.OutputFactor, snap.Tolerance) {
		return ErrInvalidState
	}
	product, err := multiply(factor, basis)
	if err != nil || !matricesEqual(product, last.AcceptedDelta, snap.Tolerance) {
		return ErrInvalidState
	}
	if matrixDifferenceNorm(last.FullDelta, last.AcceptedDelta) > math.Max(1e-8, snap.Tolerance)*math.Max(1, frobeniusNorm(last.FullDelta)) && last.TruncationError == 0 {
		return ErrInvalidState
	}
	return nil
}

func mergeFromRaw(raw *rawMerge, outputDim, inputDim int) (*MergeResult, error) {
	if raw == nil {
		return nil, nil
	}
	if raw.CompletedRound <= 0 || raw.BasisVersionBefore <= 0 || raw.PreviousRank <= 0 || raw.PreviousRank > MaxDimension || raw.RequestedRank <= 0 || raw.RequestedRank > raw.PreviousRank || raw.NewRank != raw.RequestedRank || raw.NewRank > MaxDimension || raw.NewRank > outputDim || raw.NewRank > inputDim || raw.NextRound != raw.CompletedRound+1 || raw.BasisVersionAfter != raw.BasisVersionBefore+1 {
		return nil, ErrInvalidState
	}
	fullDelta, err := fromRawMatrix(raw.FullDelta, outputDim, inputDim)
	if err != nil {
		return nil, errors.Join(ErrInvalidState, err)
	}
	accepted, err := fromRawMatrix(raw.AcceptedDelta, outputDim, inputDim)
	if err != nil {
		return nil, errors.Join(ErrInvalidState, err)
	}
	basis, err := fromRawMatrix(raw.InputBasis, raw.NewRank, inputDim)
	if err != nil {
		return nil, errors.Join(ErrInvalidState, err)
	}
	factor, err := fromRawMatrix(raw.OutputFactor, outputDim, raw.NewRank)
	if err != nil {
		return nil, errors.Join(ErrInvalidState, err)
	}
	if !isFinite(raw.TruncationError) || raw.TruncationError < 0 || !isFinite(raw.RelativeTruncationError) || raw.RelativeTruncationError < 0 || !isFinite(raw.ProductResidual) || raw.ProductResidual < 0 || !isFinite(raw.OrthogonalityError) || raw.OrthogonalityError < 0 {
		return nil, ErrInvalidState
	}
	expectedTruncated := raw.NewRank < raw.PreviousRank
	if raw.Truncated != expectedTruncated || raw.RequestedRank != raw.NewRank || (raw.Truncated && raw.Equivalent) {
		return nil, ErrInvalidState
	}
	if !raw.Truncated && !raw.Equivalent {
		return nil, ErrInvalidState
	}
	for _, value := range raw.SingularValues {
		if !isFinite(value) || value < 0 {
			return nil, ErrInvalidState
		}
	}
	product, err := multiply(factor, basis)
	if err != nil || !matricesEqual(product, accepted, 1e-8) {
		return nil, ErrInvalidState
	}
	fullNorm := frobeniusNorm(fullDelta)
	residual := matrixDifferenceNorm(fullDelta, accepted)
	if raw.Truncated {
		truncation := matrixDifferenceNorm(fullDelta, accepted)
		if math.Abs(raw.TruncationError-truncation) > 1e-8*math.Max(1, fullNorm) || math.Abs(raw.ProductResidual-truncation) > 1e-8*math.Max(1, fullNorm) {
			return nil, ErrInvalidState
		}
	} else if residual > 1e-8*math.Max(1, fullNorm) || raw.TruncationError != 0 {
		return nil, ErrInvalidState
	}
	relative := 0.0
	if fullNorm != 0 {
		relative = residual / fullNorm
	}
	if math.Abs(raw.RelativeTruncationError-relative) > 1e-8*math.Max(1, relative) {
		return nil, ErrInvalidState
	}
	return &MergeResult{
		CompletedRound:          raw.CompletedRound,
		NextRound:               raw.NextRound,
		BasisVersionBefore:      raw.BasisVersionBefore,
		BasisVersionAfter:       raw.BasisVersionAfter,
		PreviousRank:            raw.PreviousRank,
		RequestedRank:           raw.RequestedRank,
		NewRank:                 raw.NewRank,
		Truncated:               raw.Truncated,
		TruncationError:         raw.TruncationError,
		RelativeTruncationError: raw.RelativeTruncationError,
		ProductResidual:         raw.ProductResidual,
		OrthogonalityError:      raw.OrthogonalityError,
		Equivalent:              raw.Equivalent,
		SingularValues:          append([]float64(nil), raw.SingularValues...),
		FullDelta:               fullDelta,
		AcceptedDelta:           accepted,
		InputBasis:              basis,
		OutputFactor:            factor,
	}, nil
}

func matrixDifferenceNorm(a, b *mat.Dense) float64 {
	difference := mat.NewDense(a.RawMatrix().Rows, a.RawMatrix().Cols, nil)
	difference.Sub(a, b)
	return frobeniusNorm(difference)
}

func toRawMatrix(value *mat.Dense) rawMatrix {
	if value == nil {
		return nil
	}
	rows, cols := value.Dims()
	result := make(rawMatrix, rows)
	for i := 0; i < rows; i++ {
		result[i] = append([]float64(nil), value.RawRowView(i)...)
		if len(result[i]) != cols {
			result[i] = make([]float64, cols)
			for j := 0; j < cols; j++ {
				result[i][j] = value.At(i, j)
			}
		}
	}
	return result
}

func fromRawMatrix(value rawMatrix, rows, cols int) (*mat.Dense, error) {
	if rows <= 0 || rows > MaxDimension || cols <= 0 || cols > MaxDimension || len(value) != rows {
		return nil, ErrInvalidState
	}
	result := mat.NewDense(rows, cols, nil)
	for i, row := range value {
		if len(row) != cols {
			return nil, ErrInvalidState
		}
		for j, v := range row {
			if !isFinite(v) {
				return nil, ErrInvalidState
			}
			result.Set(i, j, v)
		}
	}
	return result, nil
}

func syncDirectory(path string) {
	dir, err := os.Open(path)
	if err != nil {
		return
	}
	_ = dir.Sync()
	_ = dir.Close()
}

func matricesEqual(a, b *mat.Dense, tolerance float64) bool {
	if a == nil || b == nil {
		return a == nil && b == nil
	}
	ar, ac := a.Dims()
	br, bc := b.Dims()
	if ar != br || ac != bc {
		return false
	}
	for i := 0; i < ar; i++ {
		for j := 0; j < ac; j++ {
			if mathAbs(a.At(i, j)-b.At(i, j)) > tolerance {
				return false
			}
		}
	}
	return true
}

func mathAbs(value float64) float64 {
	return math.Abs(value)
}
