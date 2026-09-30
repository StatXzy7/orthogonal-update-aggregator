package aggregator

import (
	"errors"
	"math"
	"os"
	"path/filepath"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestRejectsInvalidSubmissions(t *testing.T) {
	agg, err := New(3, 4, 2, identityRows(2, 4), WithTolerance(1e-9))
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	factor := mat.NewDense(3, 2, nil)
	factor.Set(0, 0, 1)

	if _, err := agg.CommitRound(0); !errors.Is(err, ErrNoSubmissions) {
		t.Fatalf("empty commit error = %v, want %v", err, ErrNoSubmissions)
	}
	if err := agg.Submit(Submission{Round: 1, BasisVersion: 1, ClientID: "a", OutputFactor: factor, Weight: 0}); !errors.Is(err, ErrInvalidWeight) {
		t.Fatalf("zero weight error = %v", err)
	}
	if err := agg.Submit(Submission{Round: 1, BasisVersion: 1, ClientID: "a", OutputFactor: factor, Weight: -1}); !errors.Is(err, ErrInvalidWeight) {
		t.Fatalf("negative weight error = %v", err)
	}
	if err := agg.Submit(Submission{Round: 1, BasisVersion: 1, ClientID: "", OutputFactor: factor, Weight: 1}); !errors.Is(err, ErrInvalidClient) {
		t.Fatalf("empty client error = %v", err)
	}
	if err := agg.Submit(Submission{Round: 1, BasisVersion: 1, ClientID: "a", OutputFactor: mat.NewDense(3, 1, nil), Weight: 1}); !errors.Is(err, ErrShapeMismatch) {
		t.Fatalf("shape error = %v", err)
	}
	nonFinite := mat.NewDense(3, 2, nil)
	nonFinite.Set(1, 1, math.NaN())
	if err := agg.Submit(Submission{Round: 1, BasisVersion: 1, ClientID: "bad", OutputFactor: nonFinite, Weight: 1}); !errors.Is(err, ErrNonFiniteValue) {
		t.Fatalf("non-finite error = %v", err)
	}
	if err := agg.Submit(Submission{Round: 1, BasisVersion: 1, ClientID: "a", OutputFactor: factor, Weight: 1}); err != nil {
		t.Fatalf("valid submit error = %v", err)
	}
	if err := agg.Submit(Submission{Round: 1, BasisVersion: 1, ClientID: "a", OutputFactor: factor, Weight: 2}); !errors.Is(err, ErrDuplicateClient) {
		t.Fatalf("duplicate error = %v", err)
	}
	if err := agg.Submit(Submission{Round: 2, BasisVersion: 1, ClientID: "b", OutputFactor: factor, Weight: 1}); !errors.Is(err, ErrRoundMismatch) {
		t.Fatalf("stale round error = %v", err)
	}
	if err := agg.Submit(Submission{Round: 1, BasisVersion: 2, ClientID: "c", OutputFactor: factor, Weight: 1}); !errors.Is(err, ErrBasisVersionMismatch) {
		t.Fatalf("stale basis error = %v", err)
	}
	if _, err := agg.CommitRound(3); !errors.Is(err, ErrInvalidTargetRank) {
		t.Fatalf("target rank error = %v", err)
	}
	if agg.Round() != 1 || agg.BasisVersion() != 1 || agg.PendingClientCount() != 1 {
		t.Fatalf("state changed after rejected operations: round=%d basis=%d pending=%d", agg.Round(), agg.BasisVersion(), agg.PendingClientCount())
	}
}

func TestFailedMergeIsAtomic(t *testing.T) {
	hadamard := denseFromRows(
		[]float64{0.5, 0.5, 0.5, 0.5},
		[]float64{0.5, -0.5, 0.5, -0.5},
		[]float64{0.5, 0.5, -0.5, -0.5},
		[]float64{0.5, -0.5, -0.5, 0.5},
	)
	agg, err := New(4, 4, 4, hadamard, WithTolerance(1e-9))
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	factor := mat.NewDense(4, 4, nil)
	for column := 0; column < 4; column++ {
		factor.Set(0, column, math.MaxFloat64)
	}
	if err := agg.Submit(Submission{Round: 1, BasisVersion: 1, ClientID: "overflow", OutputFactor: factor, Weight: 1}); err != nil {
		t.Fatalf("Submit() error = %v", err)
	}
	oldBasis := agg.InputBasis()
	oldFactor := agg.OutputFactor()
	if _, err := agg.CommitRound(0); !errors.Is(err, ErrNonFiniteValue) {
		t.Fatalf("overflow merge error = %v, want %v", err, ErrNonFiniteValue)
	}
	if agg.Round() != 1 || agg.BasisVersion() != 1 || agg.PendingClientCount() != 1 {
		t.Fatalf("state advanced after failed merge: round=%d basis=%d pending=%d", agg.Round(), agg.BasisVersion(), agg.PendingClientCount())
	}
	if !matrixClose(agg.InputBasis(), oldBasis, 0) || !matrixClose(agg.OutputFactor(), oldFactor, 0) {
		t.Fatalf("original basis or factor changed")
	}
	if _, err := agg.MergedIncrement(); !errors.Is(err, ErrNoMergedIncrement) {
		t.Fatalf("MergedIncrement after failed merge error = %v", err)
	}
	if clients := agg.PendingClients(); len(clients) != 1 || clients[0] != "overflow" {
		t.Fatalf("pending clients = %v", clients)
	}
	validFactor := mat.NewDense(4, 4, nil)
	validFactor.Set(0, 0, 1)
	agg.pending = map[string]*submitted{"valid": {
		Round:        1,
		BasisVersion: 1,
		ClientID:     "valid",
		Weight:       1,
		OutputFactor: validFactor,
	}}
	if _, err := agg.CommitRound(0); err != nil {
		t.Fatalf("retry after retained round error = %v", err)
	}
}

func TestSaveRestoreConsistencyWithPendingAndAfterRound(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "state.json")
	agg, err := New(3, 4, 2, identityRows(2, 4), WithTolerance(1e-9))
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	factor := denseFromRows(
		[]float64{1, 2},
		[]float64{3, 0},
		[]float64{-1, 1},
	)
	if err := agg.Submit(Submission{Round: 1, BasisVersion: 1, ClientID: "a", OutputFactor: factor, Weight: 3}); err != nil {
		t.Fatalf("Submit() error = %v", err)
	}
	if err := agg.Save(path); err != nil {
		t.Fatalf("Save(pending) error = %v", err)
	}
	restored, err := Load(path)
	if err != nil {
		t.Fatalf("Load(pending) error = %v", err)
	}
	if restored.Round() != 1 || restored.BasisVersion() != 1 || restored.PendingClientCount() != 1 {
		t.Fatalf("restored pending state round=%d basis=%d pending=%d", restored.Round(), restored.BasisVersion(), restored.PendingClientCount())
	}
	if err := restored.Submit(Submission{Round: 1, BasisVersion: 1, ClientID: "a", OutputFactor: factor, Weight: 1}); !errors.Is(err, ErrDuplicateClient) {
		t.Fatalf("restored duplicate error = %v", err)
	}
	second := denseFromRows(
		[]float64{-1, 0},
		[]float64{2, 1},
		[]float64{0, 4},
	)
	if err := restored.Submit(Submission{Round: 1, BasisVersion: 1, ClientID: "b", OutputFactor: second, Weight: 1}); err != nil {
		t.Fatalf("restored Submit(b) error = %v", err)
	}
	result, err := restored.CommitRound(0)
	if err != nil {
		t.Fatalf("restored CommitRound() error = %v", err)
	}
	if err := restored.Save(path); err != nil {
		t.Fatalf("Save(merged) error = %v", err)
	}
	final, err := Load(path)
	if err != nil {
		t.Fatalf("Load(merged) error = %v", err)
	}
	if final.Round() != 2 || final.BasisVersion() != 2 || final.PendingClientCount() != 0 {
		t.Fatalf("final state round=%d basis=%d pending=%d", final.Round(), final.BasisVersion(), final.PendingClientCount())
	}
	current, err := final.CurrentIncrement()
	if err != nil {
		t.Fatalf("final CurrentIncrement() error = %v", err)
	}
	if !matrixClose(current, result.FullDelta, 1e-12) {
		t.Fatalf("restored product = %v\nwant %v", formatted(current), formatted(result.FullDelta))
	}
	queried, err := final.MergedIncrement()
	if err != nil {
		t.Fatalf("final MergedIncrement() error = %v", err)
	}
	if !matrixClose(queried, result.FullDelta, 1e-12) {
		t.Fatalf("restored merged delta = %v\nwant %v", formatted(queried), formatted(result.FullDelta))
	}
	next := denseFromRows(
		[]float64{0, 1},
		[]float64{1, 0},
		[]float64{2, -2},
	)
	if err := final.Submit(Submission{Round: 2, BasisVersion: 2, ClientID: "next", OutputFactor: next, Weight: 1}); err != nil {
		t.Fatalf("next-round Submit() error = %v", err)
	}
	if _, err := final.CommitRound(0); err != nil {
		t.Fatalf("next-round CommitRound() error = %v", err)
	}
}

func TestDimensionAndInitialBasisValidation(t *testing.T) {
	if _, err := New(13, 4, 2, identityRows(2, 4)); !errors.Is(err, ErrInvalidDimension) {
		t.Fatalf("large dimension error = %v", err)
	}
	if _, err := New(3, 4, 5, identityRows(2, 4)); !errors.Is(err, ErrInvalidDimension) {
		t.Fatalf("rank above input dimension error = %v", err)
	}
	if _, err := New(3, 4, 2, denseFromRows([]float64{1, 1, 0, 0}, []float64{0, 0, 1, 0})); !errors.Is(err, ErrInvalidBasis) {
		t.Fatalf("non-orthogonal basis error not rejected")
	}
	bad := filepath.Join(t.TempDir(), "bad.json")
	if err := os.WriteFile(bad, []byte("{bad"), 0o600); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}
	if _, err := Load(bad); !errors.Is(err, ErrInvalidState) {
		t.Fatalf("corrupt state error = %v", err)
	}
}
