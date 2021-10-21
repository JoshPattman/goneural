package goneural

type Layer interface{
	PropagateValues(x []float64) []float64
	GetNumInputs() int
	GetNumOutputs() int
	GetActivation() Activation
}

type GradientDescentLayer interface{
	Layer
	TrainGradientDescent(learningRate float64, layerInputs, deltas []float64) []float64
}