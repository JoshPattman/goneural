package goneural

// Activation : An interface type that denotes an activations function.
type Activation interface{
	// Calc : Calculate the activation for x
	Calc(x float64) float64
	// Diff : Calculate the differential of the activation for x
	Diff(x float64) float64
}