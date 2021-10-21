package goneural

import (
	"math/rand"
)

type DenseLayer struct{
	Weights [][]float64
	Ac Activation
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
		Ac: activation,
	}
}

func (l *DenseLayer) PropagateValues (X []float64) []float64{
	net := valsMulWeights(append(X, 1), l.Weights)
	for i := range net{
		net[i] = l.Ac.Calc(net[i])
	}
	return net
}

func (l *DenseLayer) GetNumInputs() int{
	return len(l.Weights)
}

func (l *DenseLayer) GetNumOutputs() int{
	return len(l.Weights[0])
}

func (l *DenseLayer) BackpropGD(learningRate float64, layerInputs, deltas []float64) []float64{
	layerInputs = append(layerInputs, 1)
	nextDeltas := make([]float64, len(layerInputs))

	for ni := range l.Weights{
		for no := range l.Weights[0]{
			nextDeltas[ni] += deltas[no] * l.Weights[ni][no]
		}
		nextDeltas[ni] = nextDeltas[ni] * l.Ac.Diff(layerInputs[ni])
	}

	for no := range deltas{
		for ni := range l.Weights{
			l.Weights[ni][no] += learningRate * deltas[no] * layerInputs[ni]
		}
	}
	// remove delta for bias as we dont want it
	return nextDeltas[:len(nextDeltas)-1]
}

func GetLasLayerDeltas(pred, expec []float64, loss LossFunction, ac Activation) []float64{
	deltas := GetLayerLossDiffs(loss, expec, pred)
	for i := range deltas{
		deltas[i] = deltas[i] * ac.Diff(pred[i])
	}
	return deltas
}