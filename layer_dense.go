package goneural

import "math/rand"

type DenseLayer struct{
	Weights [][]float64
	Ac Activation
}

func NewDenseLayer(numInputs, numOutputs int, activation Activation) *DenseLayer{
	W := make([][]float64, numInputs)
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
	net := valsMulWeights(X, l.Weights)
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

func (l *DenseLayer) BackpropagateOutputLayer(loss LossFunction, predictedInputs, expected, predicted []float64){
	for oi := range l.Weights[0]{
		error := loss.SingleLossDiff(expected[oi], predicted[oi]) * l.Ac.Diff(predicted[oi])
		for ii := range l.Weights{
			l.Weights[ii][oi] += error * predictedInputs[ii]
		}
	}
}