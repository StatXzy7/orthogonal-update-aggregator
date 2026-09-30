package aggregator

import (
	"errors"
	"math"
	"path/filepath"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func newDense(data [][]float64) *mat.Dense {
	rows := len(data)
	cols := len(data[0])
	flat := make([]float64, 0, rows*cols)
	for _, row := range data {
		if len(row) != cols {
			panic("ragged matrix")
		}
		flat = append(flat, row...)
	}
	return mat.NewDense(rows, cols, flat)
}

func submitHelper(t *testing.T, m *Merger, id string, out *mat.Dense, weight float64) {
	t.Helper()
	if err := m.Submit(Submission{
		ClientID:     id,
		Round:        m.Round(),
		BasisVersion: m.BasisVersion(),
		Out:          out,
		PosWeight:    weight,
	}); err != nil {
		t.Fatalf("Submit(%s): %v", id, err)
	}
}

func matrixApprox(t *testing.T, got, want mat.Matrix, tolerance float64, name string) {
	t.Helper()
	gr, gc := got.Dims()
	wr, wc := want.Dims()
	if gr != wr || gc != wc {
		t.Fatalf("%s dimensions = %dx%d, want %dx%d", name, gr, gc, wr, wc)
	}
	for i := 0; i < gr; i++ {
		for j := 0; j < gc; j++ {
			if math.Abs(got.At(i, j)-want.At(i, j)) > tolerance {
				t.Fatalf("%s[%d,%d] = %.12f, want %.12f", name, i, j, got.At(i, j), want.At(i, j))
			}
		}
	}
}

func TestWeightedMergeRebasisAndContinues(t *testing.T) {
	basis := newDense([][]float64{
		{3.0 / 5, 0},
		{4.0 / 5, 0},
		{0, 1},
	})
	m, err := NewMerger(3, 2, basis)
	if err != nil {
		t.Fatal(err)
	}

	x1 := newDense([][]float64{
		{1, 2},
		{-1, 0},
	})
	x2 := newDense([][]float64{
		{4, -1},
		{2, 3},
	})
	submitHelper(t, m, "a", x1, 3)
	submitHelper(t, m, "b", x2, 1)

	weighted := mat.NewDense(2, 2, nil)
	weighted.
		Scale(0.75, x1)
	second := mat.NewDense(2, 2, nil)
	second.Scale(0.25, x2)
	weighted.Add(weighted, second)
	wantDelta := mat.NewDense(2, 3, nil)
	wantDelta.Mul(weighted, basis.T())

	merged, err := m.MergeDefault()
	if err != nil {
		t.Fatalf("MergeDefault: %v", err)
	}
	if m.Round() != 2 || m.BasisVersion() != 2 || m.PendingCount() != 0 {
		t.Fatalf("state after merge = round %d basis %d pending %d", m.Round(), m.BasisVersion(), m.PendingCount())
	}
	matrixApprox(t, merged.Delta, wantDelta, 1e-12, "weighted delta")
	if math.Abs(merged.TotalWeight-4) > 1e-15 || merged.NumClients != 2 || !merged.ExactEquivalence {
		t.Fatalf("merge metadata = weight %v clients %d exact %v", merged.TotalWeight, merged.NumClients, merged.ExactEquivalence)
	}
	if merged.Truncated || merged.TruncationError > 1e-11 || merged.ProductError > 1e-11 {
		t.Fatalf("default merge claimed truncation: err %g tail %g", merged.ProductError, merged.TruncationError)
	}
	product := mat.NewDense(2, 3, nil)
	product.Mul(merged.Out, merged.In)
	matrixApprox(t, product, wantDelta, 1e-10, "Out*In")
	if frobeniusDiff(merged.Basis, basis) < 1e-8 {
		t.Fatal("new round should use the refactorized basis, not the client input basis")
	}
	if frobeniusDiff(merged.Basis, m.Basis()) > 1e-14 || frobeniusDiff(merged.Basis.T(), merged.In) > 1e-14 {
		t.Fatal("returned basis and merger basis disagree")
	}
	if !orthonormalColumns(merged.Basis, 1e-10) || !orthonormalRows(merged.In, 1e-10) {
		t.Fatal("new basis is not orthonormal")
	}

	nextOut := newDense([][]float64{
		{0.5, -1.25},
		{2, 0.5},
	})
	nextBasis := m.Basis()
	submitHelper(t, m, "c", nextOut, 2)
	merged2, err := m.MergeDefault()
	if err != nil {
		t.Fatalf("second MergeDefault: %v", err)
	}
	wantDelta2 := mat.NewDense(2, 3, nil)
	wantDelta2.Mul(nextOut, nextBasis.T())
	if m.Round() != 3 || merged2.Round != 2 {
		t.Fatalf("round progression = %d / merged %d", m.Round(), merged2.Round)
	}
	product2 := mat.NewDense(2, 3, nil)
	product2.Mul(merged2.Out, merged2.In)
	matrixApprox(t, product2, wantDelta2, 1e-10, "second round product")
}

func TestTruncationReportsError(t *testing.T) {
	m, err := NewMerger(2, 2, nil)
	if err != nil {
		t.Fatal(err)
	}
	out := newDense([][]float64{
		{3, 1},
		{-3, 1},
	})
	submitHelper(t, m, "only", out, 1)
	merged, err := m.Merge(1)
	if err != nil {
		t.Fatalf("Merge(1): %v", err)
	}
	if m.Rank() != 1 || !merged.RankReduced || !merged.Truncated || merged.ExactEquivalence {
		t.Fatalf("rank=%d reduced=%v truncated=%v exact=%v", m.Rank(), merged.RankReduced, merged.Truncated, merged.ExactEquivalence)
	}
	if math.Abs(merged.TruncationError-math.Sqrt2) > 1e-10 {
		t.Fatalf("truncation error = %.12f, want %.12f", merged.TruncationError, math.Sqrt2)
	}
	if math.Abs(merged.TruncationError-merged.ProductError) > 1e-10 {
		t.Fatalf("product error %.12f != tail %.12f", merged.ProductError, merged.TruncationError)
	}
	if len(merged.SingularValues) != 1 || merged.SingularValues[0] <= 0 {
		t.Fatalf("singular values = %v", merged.SingularValues)
	}
}

func TestZeroAndRepeatedSingularValuesAreFinite(t *testing.T) {
	m, err := NewMerger(2, 2, nil)
	if err != nil {
		t.Fatal(err)
	}
	submitHelper(t, m, "zero", mat.NewDense(2, 2, nil), 1)
	merged, err := m.MergeDefault()
	if err != nil {
		t.Fatalf("zero merge: %v", err)
	}
	if !isFiniteDense(merged.Out) || !isFiniteDense(merged.In) {
		t.Fatal("zero delta produced non-finite factors")
	}
	if !orthonormalRows(merged.In, 1e-10) || merged.ProductError != 0 {
		t.Fatal("zero delta did not preserve an orthonormal basis")
	}

	m2, err := NewMerger(2, 2, nil)
	if err != nil {
		t.Fatal(err)
	}
	repeated := newDense([][]float64{
		{2, 0},
		{0, 2},
	})
	submitHelper(t, m2, "repeat", repeated, 1)
	merged2, err := m2.MergeDefault()
	if err != nil {
		t.Fatalf("repeated singular merge: %v", err)
	}
	for _, sigma := range merged2.SingularValues {
		if math.Abs(sigma-2) > 1e-10 {
			t.Fatalf("singular value = %.12f, want 2", sigma)
		}
	}
	product := mat.NewDense(2, 2, nil)
	product.Mul(merged2.Out, merged2.In)
	matrixApprox(t, product, repeated, 1e-10, "repeated-sigma product")
}

func TestRejectsInvalidSubmissions(t *testing.T) {
	m, err := NewMerger(2, 2, nil)
	if err != nil {
		t.Fatal(err)
	}
	valid := mat.NewDense(1, 2, []float64{1, 1})
	submitHelper(t, m, "same", valid, 1)

	tests := []struct {
		name       string
		submission Submission
		want       error
	}{
		{"empty id", Submission{Round: 1, BasisVersion: 1, Out: valid, PosWeight: 1}, ErrEmptyClientID},
		{"duplicate", Submission{ClientID: "same", Round: 1, BasisVersion: 1, Out: valid, PosWeight: 1}, ErrDuplicateClient},
		{"stale round", Submission{ClientID: "old", Round: 2, BasisVersion: 1, Out: valid, PosWeight: 1}, ErrStaleRound},
		{"old basis", Submission{ClientID: "old-basis", Round: 1, BasisVersion: 2, Out: valid, PosWeight: 1}, ErrBasisMismatch},
		{"zero weight", Submission{ClientID: "zero", Round: 1, BasisVersion: 1, Out: valid, PosWeight: 0}, ErrInvalidWeight},
		{"negative weight", Submission{ClientID: "negative", Round: 1, BasisVersion: 1, Out: valid, PosWeight: -1}, ErrInvalidWeight},
		{"nan weight", Submission{ClientID: "nan", Round: 1, BasisVersion: 1, Out: valid, PosWeight: math.NaN()}, ErrInvalidWeight},
		{"wrong columns", Submission{ClientID: "shape", Round: 1, BasisVersion: 1, Out: mat.NewDense(1, 1, []float64{1}), PosWeight: 1}, ErrDimensionConflict},
		{"output mismatch", Submission{ClientID: "rows", Round: 1, BasisVersion: 1, Out: mat.NewDense(2, 2, nil), PosWeight: 1}, ErrDimensionConflict},
		{"non-finite", Submission{ClientID: "inf", Round: 1, BasisVersion: 1, Out: mat.NewDense(1, 2, []float64{math.Inf(1), 0}), PosWeight: 1}, ErrNonFinite},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := m.Submit(tt.submission); !errors.Is(err, tt.want) {
				t.Fatalf("Submit error = %v, want %v", err, tt.want)
			}
		})
	}
	if count := m.PendingCount(); count != 1 {
		t.Fatalf("invalid submissions changed pending count to %d", count)
	}
}

func TestMergeValidationAndDimensions(t *testing.T) {
	m, err := NewMerger(2, 2, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := m.MergeDefault(); !errors.Is(err, ErrNoClients) {
		t.Fatalf("empty merge error = %v, want %v", err, ErrNoClients)
	}
	submitHelper(t, m, "a", mat.NewDense(1, 2, nil), 1)
	for _, rank := range []int{0, -1, 3} {
		if _, err := m.Merge(rank); !errors.Is(err, ErrRankOutOfRange) {
			t.Fatalf("Merge(%d) error = %v, want rank error", rank, err)
		}
	}
	if m.Round() != 1 || m.BasisVersion() != 1 || m.PendingCount() != 1 {
		t.Fatal("failed rank validation changed merger state")
	}

	large := mat.NewDense(MaxDimension+1, 1, nil)
	largeSubmission := Submission{ClientID: "large", Round: 1, BasisVersion: 1, Out: large, PosWeight: 1}
	if err := m.Submit(largeSubmission); !errors.Is(err, ErrDimensionTooLarge) {
		t.Fatalf("large submission error = %v, want %v", err, ErrDimensionTooLarge)
	}

	badBasis := mat.NewDense(2, 2, []float64{1, 1, 0, 0})
	if _, err := NewMerger(2, 2, badBasis); !errors.Is(err, ErrNotOrthonormal) {
		t.Fatalf("bad basis error = %v, want %v", err, ErrNotOrthonormal)
	}
	if _, err := NewMerger(13, 1, nil); !errors.Is(err, ErrDimensionTooLarge) {
		t.Fatalf("large input error = %v, want %v", err, ErrDimensionTooLarge)
	}
}

func TestMergeFailureIsAtomic(t *testing.T) {
	m, err := NewMerger(2, 2, nil)
	if err != nil {
		t.Fatal(err)
	}
	out1 := mat.NewDense(2, 2, []float64{1, 2, 3, 4})
	out2 := mat.NewDense(2, 2, []float64{-1, 0, 2, 1})
	submitHelper(t, m, "a", out1, 1)
	submitHelper(t, m, "b", out2, 2)

	oldBasis := m.Basis()
	oldRound := m.Round()
	oldVersion := m.BasisVersion()
	oldRank := m.Rank()
	oldPending := m.Pending()

	originalFactorizer := factorizeDelta
	injected := errors.New("injected svd failure")
	factorizeDelta = func(*mat.Dense, int) (*mat.Dense, *mat.Dense, []float64, float64, error) {
		return nil, nil, nil, 0, injected
	}
	_, mergeErr := m.MergeDefault()
	factorizeDelta = originalFactorizer
	if !errors.Is(mergeErr, injected) || !errors.Is(mergeErr, ErrNumericalFailure) {
		t.Fatalf("Merge error = %v, want injected numerical failure", mergeErr)
	}

	if m.Round() != oldRound || m.BasisVersion() != oldVersion || m.Rank() != oldRank {
		t.Fatal("failure changed round, basis version, or rank")
	}
	if diff := frobeniusDiff(m.Basis(), oldBasis); diff != 0 {
		t.Fatalf("failure changed basis by %.12f", diff)
	}
	pending := m.Pending()
	if len(pending) != len(oldPending) {
		t.Fatalf("pending count = %d, want %d", len(pending), len(oldPending))
	}
	for i := range pending {
		if pending[i].ClientID != oldPending[i].ClientID || pending[i].Round != oldPending[i].Round ||
			pending[i].BasisVersion != oldPending[i].BasisVersion ||
			math.Abs(pending[i].PosWeight-oldPending[i].PosWeight) > 0 {
			t.Fatal("failure changed pending submission metadata")
		}
		if diff := frobeniusDiff(pending[i].Out, oldPending[i].Out); diff != 0 {
			t.Fatalf("failure changed pending factor for client %s", pending[i].ClientID)
		}
	}
	if err := m.Submit(Submission{ClientID: "a", Round: m.Round(), BasisVersion: m.BasisVersion(), Out: out1, PosWeight: 1}); !errors.Is(err, ErrDuplicateClient) {
		t.Fatalf("duplicate after failed merge error = %v", err)
	}

	merged, err := m.MergeDefault()
	if err != nil {
		t.Fatalf("retry after failed merge: %v", err)
	}
	if m.Round() != oldRound+1 || m.PendingCount() != 0 {
		t.Fatal("successful retry did not advance and clear pending submissions")
	}
	if productError := merged.ProductError; productError > 1e-10 {
		t.Fatalf("retry product error = %.12f", productError)
	}
}

func TestSaveAndLoadContinuesRound(t *testing.T) {
	m, err := NewMerger(3, 2, nil)
	if err != nil {
		t.Fatal(err)
	}
	first := newDense([][]float64{
		{1, -1},
		{0, 2},
	})
	submitHelper(t, m, "a", first, 2)
	pendingPath := filepath.Join(t.TempDir(), "pending.json")
	if err := m.Save(pendingPath); err != nil {
		t.Fatalf("Save pending: %v", err)
	}
	restoredPending, err := Load(pendingPath)
	if err != nil {
		t.Fatalf("Load pending: %v", err)
	}
	if restoredPending.Round() != 1 || restoredPending.BasisVersion() != 1 || restoredPending.PendingCount() != 1 {
		t.Fatal("restored pending state does not match")
	}
	if err := restoredPending.Submit(Submission{
		ClientID:     "a",
		Round:        1,
		BasisVersion: 1,
		Out:          first,
		PosWeight:    2,
	}); !errors.Is(err, ErrDuplicateClient) {
		t.Fatalf("restored duplicate error = %v", err)
	}
	second := newDense([][]float64{
		{3, 1},
		{-2, 2},
	})
	submitHelper(t, restoredPending, "b", second, 1)
	firstMerged, err := restoredPending.MergeDefault()
	if err != nil {
		t.Fatalf("merge restored pending: %v", err)
	}

	nextOut := newDense([][]float64{
		{0.5, 0.5},
		{-1, 1},
	})
	submitHelper(t, restoredPending, "c", nextOut, 3)
	mergedPath := filepath.Join(t.TempDir(), "merged.json")
	if err := restoredPending.Save(mergedPath); err != nil {
		t.Fatalf("Save merged: %v", err)
	}
	restored, err := Load(mergedPath)
	if err != nil {
		t.Fatalf("Load merged: %v", err)
	}
	if restored.Round() != 2 || restored.BasisVersion() != 2 || restored.Rank() != 2 || restored.PendingCount() != 1 {
		t.Fatalf("restored state = round %d basis %d rank %d pending %d", restored.Round(), restored.BasisVersion(), restored.Rank(), restored.PendingCount())
	}
	if diff := frobeniusDiff(restored.Basis(), restoredPending.Basis()); diff > 1e-13 {
		t.Fatalf("restored basis differs by %.12f", diff)
	}
	last, ok := restored.LastMerged()
	if !ok {
		t.Fatal("restored state is missing last merged delta")
	}
	if diff := frobeniusDiff(last.Delta, firstMerged.Delta); diff > 1e-13 {
		t.Fatal("restored last merged delta does not match")
	}
	if err := restored.Submit(Submission{ClientID: "stale", Round: 1, BasisVersion: 1, Out: nextOut, PosWeight: 1}); !errors.Is(err, ErrStaleRound) {
		t.Fatalf("stale restored submission error = %v", err)
	}

	nextBasis := restored.Basis()
	secondMerged, err := restored.MergeDefault()
	if err != nil {
		t.Fatalf("merge after restore: %v", err)
	}
	wantDelta := mat.NewDense(2, 3, nil)
	wantDelta.Mul(nextOut, nextBasis.T())
	product := mat.NewDense(2, 3, nil)
	product.Mul(secondMerged.Out, secondMerged.In)
	matrixApprox(t, product, wantDelta, 1e-10, "restored continuation delta")
	if restored.Round() != 3 || restored.BasisVersion() != 3 {
		t.Fatal("restored merger did not continue to the next round")
	}
}

func TestReturnedAndSubmittedMatricesAreCopies(t *testing.T) {
	m, err := NewMerger(2, 2, nil)
	if err != nil {
		t.Fatal(err)
	}
	out := mat.NewDense(1, 2, []float64{1, 2})
	submitHelper(t, m, "a", out, 1)
	out.Set(0, 0, 99)
	pending := m.Pending()
	if got := pending[0].Out.At(0, 0); got != 1 {
		t.Fatalf("submission aliased caller mutation, got %.12f", got)
	}
	pending[0].Out.Set(0, 0, 77)
	if got := m.Pending()[0].Out.At(0, 0); got != 1 {
		t.Fatalf("pending query aliased internal submission, got %.12f", got)
	}

	basis := m.Basis()
	basis.Set(0, 0, 3)
	if got := m.Basis().At(0, 0); got != 1 {
		t.Fatalf("basis query aliased internal basis, got %.12f", got)
	}

	merged, err := m.MergeDefault()
	if err != nil {
		t.Fatal(err)
	}
	merged.Delta.Set(0, 0, 123)
	last, ok := m.LastMerged()
	if !ok || last.Delta.At(0, 0) == 123 {
		t.Fatal("merged result aliases internal last delta")
	}
}
