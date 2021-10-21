package goneural

import (
	"go/types"
	"math"
)

type SigmoidActivation types.Nil

func (a SigmoidActivation) Calc(x float64) float64{
	return 1.0 / (1.0+math.Pow(math.E, -x))
}

func (a SigmoidActivation) Diff(x float64)float64{
	//return x * (1.0-x)
	sx := a.Calc(x)
	return sx * (1.0-sx)
}