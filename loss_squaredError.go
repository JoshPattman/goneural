+package goneural

import (
	"go/types"
	"math"
)

type SquaredErrorLoss types.Nil

func (l SquaredErrorLoss) SingleLoss(expected, predicted float64) float64{
	return 0.5*math.Pow(expected - predicted, 2)
}

func (l SquaredErrorLoss) SingleLossDiff(expected, predicted float64) float64{
	return expected - predicted
}