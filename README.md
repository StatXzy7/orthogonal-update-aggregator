# orthogonal-update-aggregator

A Linux/CPU Go implementation of a low-rank incremental update aggregator.

All clients in a round use the same `n × r` common basis `B`. Each client
submits only an `m × r` output-side factor `X_i` and a positive finite
positive-sample weight `w_i`. The server first reconstructs the true complete
update:

```text
D = (Σ_i w_i X_i / Σ_i w_i) Bᵀ
```

The output factors are combined only as part of this matrix product; the
implementation never averages independent output/input factor pairs.

`D` is then factorized with a full local SVD:

```text
D = U Σ Vᵀ = Out In
```

- `Out` has shape `m × k`.
- `In = V_kᵀ` has shape `k × n` and orthonormal rows.
- `Inᵀ` is the fresh `n × k` common basis for the next round.

The default target rank is the current rank. Product equivalence is checked in
Frobenius norm and orthogonality is checked directly. SVD singular-vector signs
are not compared. A lower target rank is allowed; `TruncationError` reports the
Frobenius norm of the discarded singular tail, and the result is explicitly not
marked exactly equivalent.

## Guarantees

- Rejects empty ID, duplicate clients in the same round, stale rounds, and
  basis-version mismatches.
- Rejects zero, negative, NaN, infinite, or overflowing weights.
- Rejects dimension conflicts, dimensions above 12, and non-finite factors.
- Rejects target ranks outside `[1, current rank]`.
- Handles zero updates and repeated singular values with finite factors.
- Commits only after all submissions and numerical checks succeed. A failed
  merge keeps the previous round, basis, rank, last result, and pending
  submissions.
- Saves state atomically to JSON. Both completed rounds and mid-round pending
  submissions can be restored and continued.

## Usage

```go
merger, _ := aggregator.NewMerger(inputDim, rank, nil)
_ = merger.Submit(aggregator.Submission{
    ClientID:     "client-1",
    Round:        merger.Round(),
    BasisVersion: merger.BasisVersion(),
    Out:          outputFactor,
    PosWeight:    positiveSamples,
})
result, err := merger.MergeDefault()
// result.Delta is the complete update; result.Out * result.In reconstructs it.
// merger now exposes the next round's common basis through merger.Basis().
```

Use `merger.Merge(lowerRank)` to intentionally truncate. Persist and resume
with:

```go
err := merger.Save("state.json")
merger, err = aggregator.Load("state.json")
```

## Development

The only numerical dependency is the local Gonum matrix/SVD package:

```text
gonum.org/v1/gonum v0.17.0
```

Run all checks:

```sh
go test ./...
```
