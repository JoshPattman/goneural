package goneural

type Layer interface{
	PropagateValues(x []float64) []float64
	GetNumInputs() int
	GetNumOutputs() int
	GetActivation() Activation
}