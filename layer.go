package goneural

type Layer interface{
	PropagateValues(X *[]float64) *[]float64
	GetNumInputs() int
	GetNumOutputs() int
	GetActivation() Activation
}