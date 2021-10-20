package goneural

type LossFunction interface{
	SingleLoss(expected, predicted float64) float64
	SingleLossDiff(expected, predicted float64) float64
}
