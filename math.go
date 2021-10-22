package goneural

// W[fromNode][toNode]
func valsMulWeights (Y *[]float64, X *[]float64, W *[][]float64){
	for t := range (*W)[0]{
		(*Y)[t] = 0
		for f := range (*W){
			(*Y)[t] += (*W)[f][t] * (*X)[f]
		}
	}
}

// W[fromNode][toNode]
// Y and X are 1D, W is 2D
func valsMulWeightsMatrix (Y *Matrix, X *Matrix, W *Matrix){
	yt := 0.0
	for t := 0; t < W.Shape[1]; t++{
		yt = 0
		for f := 0; f < W.Shape[0]; f++{
			yt += W.GetValue2D(f, t) * X.GetValue1D(f)
		}
		Y.SetValue1D(t, yt)
	}
}
