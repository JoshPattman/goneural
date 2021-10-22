package goneural


type GradientDescentOptimizer struct{
	LearningRate float64
}

type GradientDescentLayer interface{
	Layer
	TrainGradientDescent(learningRate float64, layerInputs, deltas *[]float64) *[]float64
}

func (o *GradientDescentOptimizer) FitOne(n *FeedForwardNetwork, X, Y []float64){
	YP := make([]*[]float64, len(n.Layers)+1)
	YP[0] = &X
	for i := range n.Layers{
		YP[i+1] = n.Layers[i].PropagateValues(YP[i])

	}
	firstDeltas := n.GetLastLayerDeltas(*YP[len(YP)-1], Y)
	deltas := &firstDeltas
	for invi := range YP{
		i := len(YP) - 1 - invi
		if i > 0 {
			//don't want to do this for inputs
			deltas = n.Layers[i-1].(GradientDescentLayer).TrainGradientDescent(o.LearningRate, YP[i-1], deltas)
		}
	}
}