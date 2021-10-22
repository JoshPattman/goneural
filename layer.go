package goneural

type Layer interface{
	PropagateValues(X *Matrix) *Matrix
	GetNumInputs() int
	GetNumOutputs() int
	GetActivation() Activation
}