package goneural

import "go/types"

// LinearActivation : Activation where y=x
type LinearActivation types.Nil

func (a LinearActivation)Calc(x float64)float64{
	return x
}

func (a LinearActivation)Diff(x float64)float64{
	return 1.0
}
