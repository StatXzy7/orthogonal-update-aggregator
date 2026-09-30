package aggregator

import (
	"errors"
	"math"
	"sort"

	"gonum.org/v1/gonum/mat"
)

func denseCopy(value Matrix) (dense *mat.Dense, err error) {
	if value == nil {
		return nil, ErrShapeMismatch
	}
	defer func() {
		if recover() != nil {
			dense = nil
			err = ErrShapeMismatch
		}
	}()
	rows, cols := value.Dims()
	if rows <= 0 || cols <= 0 {
		return nil, ErrInvalidDimension
	}
	if rows > MaxDimension || cols > MaxDimension {
		return nil, ErrDimensionLimit
	}
	dense = mat.NewDense(rows, cols, nil)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			v := value.At(i, j)
			if !isFinite(v) {
				return nil, ErrNonFiniteValue
			}
			dense.Set(i, j, v)
		}
	}
	return dense, nil
}

func denseClone(value *mat.Dense) *mat.Dense {
	if value == nil {
		return nil
	}
	rows, cols := value.Dims()
	result := mat.NewDense(rows, cols, nil)
	result.Copy(value)
	return result
}

func validateFinite(value *mat.Dense) error {
	if value == nil {
		return ErrShapeMismatch
	}
	raw := value.RawMatrix()
	for _, v := range raw.Data {
		if !isFinite(v) {
			return ErrNonFiniteValue
		}
	}
	return nil
}

func isFinite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func multiply(a, b *mat.Dense) (*mat.Dense, error) {
	if a == nil || b == nil {
		return nil, ErrShapeMismatch
	}
	ar, ac := a.Dims()
	br, bc := b.Dims()
	if ac != br {
		return nil, ErrShapeMismatch
	}
	result := mat.NewDense(ar, bc, nil)
	result.Mul(a, b)
	if err := validateFinite(result); err != nil {
		return nil, err
	}
	return result, nil
}

func frobeniusNorm(value *mat.Dense) float64 {
	if value == nil {
		return 0
	}
	raw := value.RawMatrix()
	sum := 0.0
	for _, v := range raw.Data {
		sum += v * v
		if !isFinite(sum) {
			return math.Inf(1)
		}
	}
	return math.Sqrt(sum)
}

func orthogonalityError(basis *mat.Dense) float64 {
	if basis == nil {
		return math.Inf(1)
	}
	rows, cols := basis.Dims()
	if rows > cols {
		return math.Inf(1)
	}
	product := mat.NewDense(rows, rows, nil)
	product.Mul(basis, basis.T())
	identity := mat.NewDense(rows, rows, nil)
	for i := 0; i < rows; i++ {
		identity.Set(i, i, 1)
	}
	product.Sub(product, identity)
	if err := validateFinite(product); err != nil {
		return math.Inf(1)
	}
	return product.Norm(2)
}

func refactorIncrement(delta *mat.Dense, outputDim, inputDim, oldRank, targetRank int, tolerance float64) (*mat.Dense, *mat.Dense, *mat.Dense, []float64, float64, float64, float64, error) {
	if delta == nil {
		return nil, nil, nil, nil, 0, 0, 0, ErrDecompositionFailed
	}
	rows, cols := delta.Dims()
	if rows != outputDim || cols != inputDim {
		return nil, nil, nil, nil, 0, 0, 0, ErrShapeMismatch
	}
	if targetRank <= 0 || targetRank > oldRank || targetRank > rows || targetRank > cols {
		return nil, nil, nil, nil, 0, 0, 0, ErrInvalidTargetRank
	}
	if !isFinite(tolerance) || tolerance <= 0 {
		return nil, nil, nil, nil, 0, 0, 0, ErrInvalidTolerance
	}

	var svd mat.SVD
	if !svd.Factorize(delta, mat.SVDThinV) {
		return nil, nil, nil, nil, 0, 0, 0, ErrDecompositionFailed
	}
	values := svd.Values(nil)
	for _, value := range values {
		if !isFinite(value) || value < 0 {
			return nil, nil, nil, nil, 0, 0, 0, ErrDecompositionFailed
		}
	}

	basisRows := make([][]float64, 0, targetRank)
	for i := 0; i < len(values) && len(basisRows) < targetRank; i++ {
		if values[i] == 0 {
			continue
		}
		vector, err := singularRightVector(&svd, cols, i, tolerance)
		if err != nil {
			return nil, nil, nil, nil, 0, 0, 0, err
		}
		basisRows = append(basisRows, vector)
	}

	for col := 0; col < cols && len(basisRows) < targetRank; col++ {
		candidate := make([]float64, cols)
		candidate[col] = 1
		candidate = projectOut(candidate, basisRows)
		norm := euclideanNorm(candidate)
		if norm > math.Sqrt(float64(cols))*tolerance {
			for i := range candidate {
				candidate[i] /= norm
			}
			basisRows = append(basisRows, candidate)
		}
	}
	if len(basisRows) != targetRank {
		return nil, nil, nil, nil, 0, 0, 0, ErrDecompositionFailed
	}

	newBasis := mat.NewDense(targetRank, cols, nil)
	for i, row := range basisRows {
		for j, value := range row {
			newBasis.Set(i, j, value)
		}
	}
	if err := validateFinite(newBasis); err != nil {
		return nil, nil, nil, nil, 0, 0, 0, errors.Join(ErrDecompositionFailed, err)
	}
	orthError := orthogonalityError(newBasis)
	if !isFinite(orthError) || orthError > tolerance {
		return nil, nil, nil, nil, 0, 0, 0, ErrEquivalenceFailed
	}

	newFactor := mat.NewDense(rows, targetRank, nil)
	newFactor.Mul(delta, newBasis.T())
	if err := validateFinite(newFactor); err != nil {
		return nil, nil, nil, nil, 0, 0, 0, errors.Join(ErrDecompositionFailed, err)
	}
	accepted, err := multiply(newFactor, newBasis)
	if err != nil {
		return nil, nil, nil, nil, 0, 0, 0, errors.Join(ErrDecompositionFailed, err)
	}
	residualMatrix := mat.NewDense(rows, cols, nil)
	residualMatrix.Sub(delta, accepted)
	residual := frobeniusNorm(residualMatrix)
	if !isFinite(residual) {
		return nil, nil, nil, nil, 0, 0, 0, ErrDecompositionFailed
	}

	truncation := 0.0
	for i := targetRank; i < len(values); i++ {
		truncation += values[i] * values[i]
	}
	truncation = math.Sqrt(truncation)
	deltaNorm := frobeniusNorm(delta)
	expected := math.Max(tolerance*math.Max(1, deltaNorm), truncation)
	if !targetRankBelowFullRank(outputDim, inputDim, oldRank, targetRank) && residual > expected {
		return nil, nil, nil, nil, 0, 0, 0, ErrEquivalenceFailed
	}
	if targetRankBelowFullRank(outputDim, inputDim, oldRank, targetRank) && math.Abs(residual-truncation) > tolerance*math.Max(1, deltaNorm) {
		return nil, nil, nil, nil, 0, 0, 0, ErrEquivalenceFailed
	}

	return newBasis, newFactor, accepted, values, residual, orthError, truncation, nil
}

func targetRankBelowFullRank(outputDim, inputDim, oldRank, targetRank int) bool {
	fullRank := min(oldRank, min(outputDim, inputDim))
	return targetRank < fullRank
}

func singularRightVector(svd *mat.SVD, cols, index int, tolerance float64) ([]float64, error) {
	thinV := mat.NewDense(cols, len(svd.Values(nil)), nil)
	svd.VTo(thinV)
	if err := validateFinite(thinV); err != nil {
		return nil, errors.Join(ErrDecompositionFailed, err)
	}
	vector := make([]float64, cols)
	for i := 0; i < cols; i++ {
		vector[i] = thinV.At(i, index)
	}
	if math.Abs(euclideanNorm(vector)-1) > tolerance {
		return nil, ErrDecompositionFailed
	}
	return vector, nil
}

func projectOut(vector []float64, basis [][]float64) []float64 {
	result := append([]float64(nil), vector...)
	for _, row := range basis {
		dot := 0.0
		for i, value := range result {
			dot += value * row[i]
		}
		for i := range result {
			result[i] -= dot * row[i]
		}
	}
	return result
}

func euclideanNorm(vector []float64) float64 {
	sum := 0.0
	for _, value := range vector {
		sum += value * value
	}
	return math.Sqrt(sum)
}

func sortStrings(values []string) {
	sort.Strings(values)
}
