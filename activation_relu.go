package goneural

import (
	"go/types"
)

type ReLUActivation types.Nil

func (a ReLUActivation) Calc(x float64) float64{
	if x > 0{
		return x
	} else{
		return 0
	}
}

func (a ReLUActivation) Diff(x float64)float64{
	if x > 0{
		return 1
	} else{
		return 0
	}
}