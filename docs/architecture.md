# Architecture Notes

## Why this shape

Research code usually fails in two ways:

1. Too flat: everything in a few giant files, hard to swap models/datasets.
2. Too fragmented: many scripts with duplicated train loops and hidden assumptions.

This template keeps one thin orchestration spine and allows complexity to grow in
`layers/`, specialized `exp/` classes, and `scripts/`.

## Extension points

1. Registry: datasets/models
2. Factory: construction policy
3. Exp hooks: batch unpacking / forward / loss / lifecycle
4. Config sections: experiment/data/model/train/eval/(project extras)

## Compatibility targets

- DLinear-style LTSF reproduction
- MillGNN-style rich model + rolling evaluation + many launch scripts
