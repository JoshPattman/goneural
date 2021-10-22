package goneural

import (
	"math/rand"
)

type DenseLayer struct{
	Weights [][]float64
	Ac   Activation
	vals []float64
}

func NewDenseLayer(numInputs, numOutputs int, activation Activation) *DenseLayer{
	W := make([][]float64, numInputs+1)
	for i := range W{
		W[i] = make([]float64, numOutputs)
		for j := range W[i]{
			W[i][j] = rand.Float64() * 2 - 1
		}
	}
	return &DenseLayer{
		Weights: W,
		Ac:      activation,
		vals:    make([]float64, len(W[0])),
	}
}

// TODO: fix this append
func (l *DenseLayer) PropagateValues (X *[]float64) *[]float64{
	withBias := append(*X, 1)
	valsMulWeights(&l.vals, &withBias, &l.Weights)
	for i := range l.vals {
		l.vals[i] = l.Ac.Calc(l.vals[i])
	}
	return &l.vals
}

func (l *DenseLayer) GetNumInputs() int{
	return len(l.Weights)-1
}

func (l *DenseLayer) GetNumOutputs() int{
	return len(l.Weights[0])
}

func (l *DenseLayer) GetActivation() Activation{
	return l.Ac
}

func (l *DenseLayer) TrainGradientDescent(learningRate float64, layerInputs, deltas *[]float64) *[]float64{
	withBias := append(*layerInputs, 1)
	layerInputs = &withBias
	nextDeltas := make([]float64, len(*layerInputs))

	for ni := range l.Weights{
		for no := range l.Weights[0]{
			nextDeltas[ni] += (*deltas)[no] * l.Weights[ni][no]
		}
		nextDeltas[ni] = nextDeltas[ni] * l.Ac.Diff((*layerInputs)[ni])
	}

	for no := range *deltas{
		for ni := range l.Weights{
			l.Weights[ni][no] += learningRate * (*deltas)[no] * (*layerInputs)[ni]
		}
	}
	// remove delta for bias as we dont want it
	clipped := nextDeltas[:len(nextDeltas)-1]
	return &clipped
}