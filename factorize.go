package aggregator

import (
	"errors"
	"math"

	"gonum.org/v1/gonum/mat"
)

var factorizeDelta = defaultFactorizeDelta

func defaultFactorizeDelta(delta *mat.Dense, targetRank int) (out, in *mat.Dense, singularValues []float64, truncationError float64, err error) {
	rows, cols := delta.Dims()
	if targetRank < 1 || targetRank > cols || rows < 1 || cols < 1 || rows > MaxDimension || cols > MaxDimension {
		return nil, nil, nil, 0, ErrRankOutOfRange
	}

	var svd mat.SVD
	if ok := svd.Factorize(delta, mat.SVDFull); !ok {
		return nil, nil, nil, 0, errors.Join(ErrNumericalFailure, errors.New("singular value decomposition failed"))
	}

	var u, v mat.Dense
	svd.UTo(&u)
	svd.VTo(&v)
	values := svd.Values(nil)

	keep := min(targetRank, len(values))
	var tailSquared float64
	for i := keep; i < len(values); i++ {
		tailSquared += values[i] * values[i]
	}
	truncationError = math.Sqrt(tailSquared)

	out = mat.NewDense(rows, targetRank, nil)
	for j := 0; j < keep; j++ {
		sigma := values[j]
		for i := 0; i < rows; i++ {
			out.Set(i, j, u.At(i, j)*sigma)
		}
	}

	// V's first targetRank columns provide an orthonormal basis even when
	// singular values repeat or are zero. Transposing them gives k×n rows.
	in = mat.NewDense(targetRank, cols, nil)
	for j := 0; j < targetRank; j++ {
		for i := 0; i < cols; i++ {
			in.Set(j, i, v.At(i, j))
		}
	}

	singularValues = make([]float64, targetRank)
	copy(singularValues, values)

	if !isFiniteDense(out) || !isFiniteDense(in) || !positiveOrZeroFinite(truncationError) {
		return nil, nil, nil, 0, errors.Join(ErrNumericalFailure, errors.New("factorization produced non-finite values"))
	}
	return out, in, singularValues, truncationError, nil
}

func frobeniusNorm(m mat.Matrix) float64 {
	rows, cols := m.Dims()
	var sum float64
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			value := m.At(i, j)
			sum += value * value
		}
	}
	return math.Sqrt(sum)
}

func frobeniusDiff(a, b mat.Matrix) float64 {
	rows, cols := a.Dims()
	bRows, bCols := b.Dims()
	if rows != bRows || cols != bCols {
		return math.Inf(1)
	}

	var sum float64
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			diff := a.At(i, j) - b.At(i, j)
			sum += diff * diff
		}
	}
	return math.Sqrt(sum)
}

func isFiniteDense(m *mat.Dense) bool {
	if m == nil {
		return false
	}
	rows, cols := m.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			value := m.At(i, j)
			if math.IsNaN(value) || math.IsInf(value, 0) {
				return false
			}
		}
	}
	return true
}

func orthonormalColumns(m *mat.Dense, tolerance float64) bool {
	rows, cols := m.Dims()
	for i := 0; i < cols; i++ {
		for j := i; j < cols; j++ {
			var dot float64
			for k := 0; k < rows; k++ {
				dot += m.At(k, i) * m.At(k, j)
			}
			want := 0.0
			if i == j {
				want = 1
			}
			if math.Abs(dot-want) > tolerance {
				return false
			}
		}
	}
	return true
}

func orthonormalRows(m *mat.Dense, tolerance float64) bool {
	rows, cols := m.Dims()
	if rows > cols {
		return false
	}
	for i := 0; i < rows; i++ {
		for j := i; j < rows; j++ {
			var dot float64
			for k := 0; k < cols; k++ {
				dot += m.At(i, k) * m.At(j, k)
			}
			want := 0.0
			if i == j {
				want = 1
			}
			if math.Abs(dot-want) > tolerance {
				return false
			}
		}
	}
	return true
}

func denseCopyOf(m *mat.Dense) *mat.Dense {
	if m == nil {
		return nil
	}
	return mat.DenseCopyOf(m)
}

func transposeCopy(m *mat.Dense) *mat.Dense {
	rows, cols := m.Dims()
	out := mat.NewDense(cols, rows, nil)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			out.Set(j, i, m.At(i, j))
		}
	}
	return out
}
