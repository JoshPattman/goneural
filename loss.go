package goneural

type LossFunction interface{
	SingleLoss(expected, predicted float64) float64
	SingleLossDiff(expected, predicted float64) float64
}

func GetLayerTotLoss(l LossFunction, expectedX, predictedX *Matrix) float64{
	if expectedX.Shape[0] !=predictedX.Shape[0]{
		panic("Expected and predicted must be same length")
	}
	out := 0.0
	for i := 0; i < expectedX.Shape[0]; i++{
		out += l.SingleLoss(expectedX.GetValue1D(i), predictedX.GetValue1D(i))
	}
	return out
}


func GetLayerLossDiffs(l LossFunction, expectedX, predictedX *Matrix) *Matrix{
	if expectedX.Shape[0] !=predictedX.Shape[0]{
		panic("Expected and predicted must be same length")
	}
	out := NewZerosMatrix(predictedX.Shape[0])
	for i := 0; i < out.Shape[0]; i++{
		out.SetValue1D(i, l.SingleLossDiff(expectedX.GetValue1D(i), predictedX.GetValue1D(i)))
	}
	return out
}