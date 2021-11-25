package goneural

import (
	"math/rand"
)

type DenseLayer struct{
	Weights Matrix
	Ac   Activation
	inputVals Matrix
	outputVals Matrix
	nextDeltas Matrix
}

func NewDenseLayer(numInputs, numOutputs int, activation Activation) *DenseLayer{
	W := make([][]float64, numInputs+1)
	for i := range W{
		W[i] = make([]float64, numOutputs)
		for j := range W[i]{
			W[i][j] = rand.Float64() * 2 - 1
		}
	}
	iv := NewZerosMatrix(numInputs+1)
	iv.SetValue1D(numInputs, 1)
	return &DenseLayer{
		Weights: *New2DMatrix(W),
		Ac:      activation,
		inputVals:    *iv,
		outputVals:    *NewZerosMatrix(numOutputs),
		nextDeltas: *NewZerosMatrix(numInputs),
	}
}

func (l *DenseLayer) PropagateValues (X *Matrix) *Matrix{
	// when we copy to this, the last value is the bias (set as 1 from creating), as inputVals is 1 longer than X
	X.CopyTo(&l.inputVals)
	valsMulWeightsMatrix(&l.outputVals, &l.inputVals, &l.Weights)
	for i := 0; i < l.outputVals.Shape[0]; i++ {
		l.outputVals.SetValue1D(i, l.Ac.Calc(l.outputVals.GetValue1D(i)))
	}
	return &l.outputVals
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

// TrainGradientDescent : DEPRECATED
func (l *DenseLayer) TrainGradientDescent(learningRate float64, layerInputs, deltas *Matrix) *Matrix{
	layerInputs.CopyTo(&l.inputVals)

	//Calculating deltas for next layer (don't bother calculating delta for bias node which is the last node)
	for ni := 0; ni < l.Weights.Shape[0]-1; ni++{
		l.nextDeltas.SetValue1D(ni, 0)
		for no := 0; no < l.Weights.Shape[1]; no++{
			d := deltas.GetValue1D(no) * l.Weights.GetValue2D(ni, no)
			l.nextDeltas.AddValue1D(ni, d)
		}
		l.nextDeltas.SetValue1D(ni, l.nextDeltas.GetValue1D(ni) * l.Ac.Diff(l.inputVals.GetValue1D(ni)))
	}

	// Updating weights
	for no := 0; no < deltas.Shape[0]; no ++{
		for ni := 0; ni < l.Weights.Shape[0]; ni ++{
			l.Weights.AddValue2D(ni, no, learningRate * deltas.GetValue1D(no) * l.inputVals.GetValue1D(ni))
		}
	}
	return &l.nextDeltas
}

func (l *DenseLayer) TrainMinibatch(learningRate float64, avInputs, avDeltas *Matrix) *Matrix{
	// this means we include biases
	avInputs.CopyTo(&l.inputVals)
	//Calculating deltas for next layer (don't bother calculating delta for bias node which is the last node)
	for ni := 0; ni < l.Weights.Shape[0]-1; ni++{
		l.nextDeltas.SetValue1D(ni, 0)
		for no := 0; no < l.Weights.Shape[1]; no++{
			d := avDeltas.GetValue1D(no) * l.Weights.GetValue2D(ni, no)
			l.nextDeltas.AddValue1D(ni, d)
		}
		l.nextDeltas.SetValue1D(ni, l.nextDeltas.GetValue1D(ni) * l.Ac.Diff(l.inputVals.GetValue1D(ni)))
	}

	// Updating weights
	for no := 0; no < avDeltas.Shape[0]; no ++{
		for ni := 0; ni < l.Weights.Shape[0]; ni ++{
			l.Weights.AddValue2D(ni, no, learningRate * avDeltas.GetValue1D(no) * l.inputVals.GetValue1D(ni))
		}
	}
	return &l.nextDeltas
}