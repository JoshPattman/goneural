package goneural

import "go/types"

type LinearActivation types.Nil

func (a *LinearActivation)Calc(x float64)float64{
	return x
}

func (a *LinearActivation)Diff(x float64)float64{
	return 1
}
