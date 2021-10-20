package goneural

type LossFunction interface{
	SingleLoss(expected, predicted float64) float64
	SingleLossDiff(expected, predicted float64) float64
}

func GetLayerLoss(l LossFunction, expectedX, predictedX []float64) []float64{
	if len(expectedX) != len(predictedX){
		panic("Expected and predicted must be same length")
	}
	outs := make([]float64, len(expectedX))
	for i := range outs{
		outs[i] = l.SingleLoss(expectedX[i], predictedX[i])
	}
	return outs
}

func GetLayerDiffLoss(l LossFunction, expectedX, predictedX []float64) []float64{
	if len(expectedX) != len(predictedX){
		panic("Expected and predicted must be same length")
	}
	outs := make([]float64, len(expectedX))
	for i := range outs{
		outs[i] = l.SingleLossDiff(expectedX[i], predictedX[i])
	}
	return outs
}