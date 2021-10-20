package goneural

// W[fromNode][toNode]
func valsMulWeights (X []float64, W [][]float64)[]float64{
	Y := make([]float64, len(W[0]))
	for t := range W[0]{
		Y[t] = 0
		for f := range W{
			Y[t] += W[f][t] * X[f]
		}
	}
	return Y
}
