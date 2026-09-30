package aggregator

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestWeightedMergeAndRebasisEquivalence(t *testing.T) {
	agg, err := New(3, 4, 2, identityRows(2, 4), WithTolerance(1e-9))
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	a := denseFromRows(
		[]float64{1, 0},
		[]float64{0, 2},
		[]float64{-1, 1},
	)
	b := denseFromRows(
		[]float64{4, 1},
		[]float64{1, 0},
		[]float64{2, -1},
	)
	if err := agg.Submit(Submission{Round: 1, BasisVersion: 1, ClientID: "a", OutputFactor: a, Weight: 0.75}); err != nil {
		t.Fatalf("Submit(a) error = %v", err)
	}
	if err := agg.Submit(Submission{Round: 1, BasisVersion: 1, ClientID: "b", OutputFactor: b, Weight: 0.25}); err != nil {
		t.Fatalf("Submit(b) error = %v", err)
	}

	result, err := agg.CommitRound(0)
	if err != nil {
		t.Fatalf("CommitRound() error = %v", err)
	}
	expectedU := denseFromRows(
		[]float64{1.75, 0.25, 0, 0},
		[]float64{0.25, 1.5, 0, 0},
		[]float64{-0.25, 0.5, 0, 0},
	)
	if !matrixClose(result.FullDelta, expectedU, 1e-12) {
		t.Fatalf("weighted delta = %v\nwant %v", formatted(result.FullDelta), formatted(expectedU))
	}
	query, err := agg.MergedIncrement()
	if err != nil {
		t.Fatalf("MergedIncrement() error = %v", err)
	}
	if !matrixClose(query, expectedU, 1e-12) {
		t.Fatalf("queried delta = %v\nwant %v", formatted(query), formatted(expectedU))
	}
	current, err := agg.CurrentIncrement()
	if err != nil {
		t.Fatalf("CurrentIncrement() error = %v", err)
	}
	if !matrixClose(current, expectedU, 1e-12) {
		t.Fatalf("rebased product = %v\nwant %v", formatted(current), formatted(expectedU))
	}
	if !result.Equivalent || result.Truncated {
		t.Fatalf("Equivalent=%v Truncated=%v", result.Equivalent, result.Truncated)
	}
	if result.ProductResidual > 1e-10 || result.OrthogonalityError > 1e-9 {
		t.Fatalf("residual=%g orth=%g", result.ProductResidual, result.OrthogonalityError)
	}
	if agg.Round() != 2 || agg.BasisVersion() != 2 || agg.PendingClientCount() != 0 {
		t.Fatalf("round=%d basis=%d pending=%d", agg.Round(), agg.BasisVersion(), agg.PendingClientCount())
	}
	if err := assertRowOrthogonal(agg.InputBasis(), 1e-9); err != nil {
		t.Fatalf("new basis: %v", err)
	}

	next := denseFromRows(
		[]float64{0, 2},
		[]float64{-1, 0},
		[]float64{1, 3},
	)
	if err := agg.Submit(Submission{Round: 2, BasisVersion: 2, ClientID: "a", OutputFactor: next, Weight: 2}); err != nil {
		t.Fatalf("next Submit() error = %v", err)
	}
	nextResult, err := agg.CommitRound(0)
	if err != nil {
		t.Fatalf("second CommitRound() error = %v", err)
	}
	expectedNext := mat.NewDense(3, 4, nil)
	expectedNext.Mul(next, result.InputBasis)
	if !matrixClose(nextResult.FullDelta, expectedNext, 1e-11) {
		t.Fatalf("second delta = %v\nwant %v", formatted(nextResult.FullDelta), formatted(expectedNext))
	}
	secondCurrent, err := agg.CurrentIncrement()
	if err != nil {
		t.Fatalf("second CurrentIncrement() error = %v", err)
	}
	if !matrixClose(secondCurrent, expectedNext, 1e-11) {
		t.Fatalf("second product = %v\nwant %v", formatted(secondCurrent), formatted(expectedNext))
	}
}

func TestTruncationReportsErrorAndNotEquivalence(t *testing.T) {
	delta := denseFromRows(
		[]float64{3, 0, 0, 0},
		[]float64{0, 2, 0, 0},
		[]float64{0, 0, 0, 0},
	)
	agg := mustNewWithDelta(t, delta, 2)
	truncationFactor := denseFromRows(
		[]float64{3, 0},
		[]float64{0, 2},
		[]float64{0, 0},
	)
	if err := agg.Submit(Submission{Round: 1, BasisVersion: 1, ClientID: "x", OutputFactor: truncationFactor, Weight: 1}); err != nil {
		t.Fatalf("Submit(delta) error = %v", err)
	}
	result, err := agg.CommitRound(1)
	if err != nil {
		t.Fatalf("CommitRound(1) error = %v", err)
	}
	if result.Equivalent || !result.Truncated {
		t.Fatalf("Equivalent=%v Truncated=%v", result.Equivalent, result.Truncated)
	}
	if math.Abs(result.TruncationError-2) > 1e-10 {
		t.Fatalf("truncation error = %g, want 2", result.TruncationError)
	}
	if math.Abs(result.RelativeTruncationError-2/math.Sqrt(13)) > 1e-10 {
		t.Fatalf("relative truncation = %g", result.RelativeTruncationError)
	}
	if !matrixClose(result.AcceptedDelta, denseFromRows(
		[]float64{3, 0, 0, 0},
		[]float64{0, 0, 0, 0},
		[]float64{0, 0, 0, 0},
	), 1e-10) {
		t.Fatalf("accepted delta = %v", formatted(result.AcceptedDelta))
	}
	if agg.Rank() != 1 {
		t.Fatalf("rank = %d", agg.Rank())
	}
}

func TestZeroIncrementAndRepeatedSingularValuesStayFinite(t *testing.T) {
	zero, err := New(3, 4, 2, identityRows(2, 4), WithTolerance(1e-9))
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	zeroFactor := mat.NewDense(3, 2, nil)
	if err := zero.Submit(Submission{Round: 1, BasisVersion: 1, ClientID: "z", OutputFactor: zeroFactor, Weight: 1}); err != nil {
		t.Fatalf("Submit(zero) error = %v", err)
	}
	zeroResult, err := zero.CommitRound(0)
	if err != nil {
		t.Fatalf("CommitRound(zero) error = %v", err)
	}
	if !allFiniteDense(zeroResult.InputBasis) || !allFiniteDense(zeroResult.OutputFactor) {
		t.Fatalf("zero decomposition produced non-finite factors")
	}
	if zeroResult.TruncationError != 0 || zeroResult.ProductResidual != 0 || !zeroResult.Equivalent {
		t.Fatalf("zero result=%+v", zeroResult)
	}

	repeatedDelta := denseFromRows(
		[]float64{1, 0, 0, 0},
		[]float64{0, 1, 0, 0},
		[]float64{0, 0, 0, 0},
	)
	repeated := mustNewWithDelta(t, repeatedDelta, 2)
	repeatedFactor := denseFromRows(
		[]float64{1, 0},
		[]float64{0, 1},
		[]float64{0, 0},
	)
	if err := repeated.Submit(Submission{Round: 1, BasisVersion: 1, ClientID: "r", OutputFactor: repeatedFactor, Weight: 1}); err != nil {
		t.Fatalf("Submit(repeated) error = %v", err)
	}
	repeatedResult, err := repeated.CommitRound(0)
	if err != nil {
		t.Fatalf("CommitRound(repeated) error = %v", err)
	}
	if !allFiniteDense(repeatedResult.InputBasis) || !allFiniteDense(repeatedResult.OutputFactor) {
		t.Fatalf("repeated-singular decomposition produced non-finite factors")
	}
	if !repeatedResult.Equivalent || repeatedResult.ProductResidual > 1e-10 {
		t.Fatalf("repeated result=%+v", repeatedResult)
	}
}
