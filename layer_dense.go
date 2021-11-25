package goneural

import (
	"math/rand"
)

// DenseLayer : The standard type of dense neural network layer
type DenseLayer struct{
	Weights Matrix
	Ac   Activation
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
		Weights: *New2DMatrix(W),
		Ac:      activation,
	}
}

func (l *DenseLayer) PropagateValues (X *Matrix) *Matrix{
	// when we copy to this, the last value is the bias (set as 1 from creating), as inputVals is 1 longer than X
	X1 := New1DMatrix(make([]float64, X.Shape[0]+1))
	X1.SetValue1D(X.Shape[0], 1)
	X.CopyTo(X1)
	Y1 := New1DMatrix(make([]float64, l.Weights.Shape[1]))
	valsMulWeightsMatrix(Y1, X1, &l.Weights)
	for i := 0; i < Y1.Shape[0]; i++ {
		Y1.SetValue1D(i, l.Ac.Calc(Y1.GetValue1D(i)))
	}
	return Y1
}

func (l *DenseLayer) GetNumInputs() int{
	// ignore the bias
	return l.Weights.Shape[0]-1
}

func (l *DenseLayer) GetNumOutputs() int{
	return l.Weights.Shape[1]
}

func (l *DenseLayer) GetActivation() Activation{
	return l.Ac
}

func (l *DenseLayer) TrainMinibatch(learningRate float64, avInputs, avDeltas *Matrix) *Matrix{
	// this means we include biases
	X1 := New1DMatrix(make([]float64, avInputs.Shape[0]+1))
	X1.SetValue1D(avInputs.Shape[0], 1)
	avInputs.CopyTo(X1)
	nextDeltas := New1DMatrix(make([]float64, avInputs.Shape[0]))
	//Calculating deltas for next layer (don't bother calculating delta for bias node which is the last node)
	for ni := 0; ni < l.Weights.Shape[0]-1; ni++{
		for no := 0; no < l.Weights.Shape[1]; no++{
			d := avDeltas.GetValue1D(no) * l.Weights.GetValue2D(ni, no)
			nextDeltas.AddValue1D(ni, d)
		}
		nextDeltas.SetValue1D(ni, nextDeltas.GetValue1D(ni) * l.Ac.Diff(X1.GetValue1D(ni)))
	}

	// Updating weights
	for no := 0; no < avDeltas.Shape[0]; no ++{
		for ni := 0; ni < l.Weights.Shape[0]; ni ++{
			l.Weights.AddValue2D(ni, no, learningRate * avDeltas.GetValue1D(no) * X1.GetValue1D(ni))
		}
	}
	return nextDeltas
}