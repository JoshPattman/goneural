package goneural

import "math"

// ActivationFunction : An activation function that is applied to neurons
type ActivationFunction func(x float64) float64

// LinearActivation : A linear activation function. This is the same as no activation function
func LinearActivation (x float64) float64{
	return x
}

// ReLUActivation : Linear when above 0 but otherwise the value is always 0
func ReLUActivation(x float64) float64{
	if x < 0{
		return 0
	}
	return x
}

// LeakyReLUActivation : Linear when above 0, but when below 0 the value is multiplied by 1/5
func LeakyReLUActivation(x float64) float64{
	if x < 0{
		return x/5
	}
	return x
}

// SigmoidActivation : A sigmoid function whose value is always between 0 and 1 (good for output layer probabilities)
func SigmoidActivation(x float64) float64{
	return 1.0/(1.0+math.Pow(2.7183, -x))
}

///////////////////////////////////// FUNCTION INDEX /////////////////////////////////////

// activationFunctions : the default activation functions mapped from a string form of their name
var activationFunctions = map[string]ActivationFunction{
	"linear":LinearActivation,
	"relu":ReLUActivation,
	"leaky_relu":LeakyReLUActivation,
	"sigmoid":SigmoidActivation,
}

// AddSerialisedActivationFunction : Adds an activation function to the index so that layers can use it
func AddSerialisedActivationFunction(name string, a ActivationFunction){
	activationFunctions[name] = a
}
