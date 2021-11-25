package goneural

// Layer : A interface type that describes layer in a feed forward network
type Layer interface{
	// PropagateValues : Propagates an input matrix though the layer and returns the calculated matrix
	PropagateValues(X *Matrix) *Matrix
	// GetNumInputs : Gets the number of inputs
	GetNumInputs() int
	// GetNumOutputs : Gets the number of outputs
	GetNumOutputs() int
	// GetActivation : Gets the activation function of this layer
	GetActivation() Activation
}