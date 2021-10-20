package goneural

import "math/rand"

type DenseLayer struct{
	Weights [][]float64
}

func NewDenseLayer(numInputs, numOutputs int) *DenseLayer{
	W := make([][]float64, numInputs)
	for i := range W{
		W[i] = make([]float64, numOutputs)
		for j := range W[i]{
			W[i][j] = rand.Float64() * 2 - 1
		}
	}
	return &DenseLayer{
		Weights: W,
	}
}

func (l *DenseLayer) PropagateValues (X []float64) []float64{
	return valsMulWeights(X, l.Weights)
}

func (l *DenseLayer) GetNumInputs() int{
	return len(l.Weights)
}

func (l *DenseLayer) GetNumOutputs() int{
	return len(l.Weights[0])
}