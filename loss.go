package goneural

type LossFunction interface{
	SingleLoss(expected, predicted float64) float64
	SingleLossDiff(expected, predicted float64) float64
}

func GetLayerTotLoss(l LossFunction, expectedX, predictedX []float64) float64{
	if len(expectedX) != len(predictedX){
		panic("Expected and predicted must be same length")
	}
	out := 0.0
	for i := range expectedX{
		out += l.SingleLoss(expectedX[i], predictedX[i])
	}
	return out
}


func GetLayerLossDiffs(l LossFunction, expectedX, predictedX []float64) []float64{
	if len(expectedX) != len(predictedX){
		panic("Expected and predicted must be same length")
	}
	out := make([]float64, len(predictedX))
	for i := range expectedX{
		out[i] = l.SingleLossDiff(expectedX[i], predictedX[i])
	}
	return out
}