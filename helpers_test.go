package aggregator

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func denseFromRows(rows ...[]float64) *mat.Dense {
	if len(rows) == 0 {
		return nil
	}
	values := make([]float64, 0, len(rows)*len(rows[0]))
	for _, row := range rows {
		values = append(values, row...)
	}
	return mat.NewDense(len(rows), len(rows[0]), values)
}

func identityRows(rows, cols int) *mat.Dense {
	result := mat.NewDense(rows, cols, nil)
	for i := 0; i < rows; i++ {
		result.Set(i, i, 1)
	}
	return result
}

func matrixClose(a, b mat.Matrix, tolerance float64) bool {
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
			if math.Abs(a.At(i, j)-b.At(i, j)) > tolerance {
				return false
			}
		}
	}
	return true
}

func formatted(value mat.Matrix) string {
	if value == nil {
		return "<nil>"
	}
	return strings.ReplaceAll(fmt.Sprintf("%v", mat.Formatted(value)), "\n", "\n\t")
}

func assertRowOrthogonal(basis *mat.Dense, tolerance float64) error {
	if orthogonalityError(basis) > tolerance {
		return fmt.Errorf("basis %v has orthogonality error %g", formatted(basis), orthogonalityError(basis))
	}
	return nil
}

func allFiniteDense(value *mat.Dense) bool {
	if value == nil {
		return false
	}
	rows, cols := value.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			v := value.At(i, j)
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return false
			}
		}
	}
	return true
}

func mustNewWithDelta(t *testing.T, delta *mat.Dense, rank int) *Aggregator {
	t.Helper()
	rows, cols := delta.Dims()
	agg, err := New(rows, cols, rank, identityRows(rank, cols), WithTolerance(1e-9))
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	return agg
}
