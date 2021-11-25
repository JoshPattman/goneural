package goneural

type SGDOptimizer struct{
	LearningRate float64
	BatchSize int
	n int
	deltas *Matrix
}


type SGDLayer interface{
	Layer
	TrainSGD(learningRate float64, layerInputs, totDeltas *Matrix, n int) *Matrix
	ResetSGD()
}

func (o *SGDOptimizer) FitOne(n *FeedForwardNetwork, X, Y *Matrix){
	YP := make([]*Matrix, len(n.Layers)+1)
	YP[0] = X
	for i := range n.Layers {
		YP[i+1] = n.Layers[i].PropagateValues(YP[i])
	}
	deltas := n.GetLastLayerDeltas(YP[len(YP)-1], Y)
	// only train when we are at batch size
	// also on first setup initialise the combined training matrix
	if o.n == 0{
		o.deltas = New1DMatrix(make([]float64, deltas.Shape[0]))
	}
	o.deltas.Add(deltas)
	if (o.n % o.BatchSize == 0) && o.n != 0 {
		o.n = 0
		for invi := range YP {
			i := len(YP) - 1 - invi
			if i > 0 {
				//don't want to do this for inputs
				deltas = n.Layers[i-1].(SGDLayer).TrainSGD(o.LearningRate, YP[i-1], o.deltas, o.BatchSize)
			}
		}
		o.deltas.Values = make([]float64, deltas.Shape[0])
		for i := range n.Layers {
			n.Layers[i].(SGDLayer).ResetSGD()
		}
	}
	o.n++
}