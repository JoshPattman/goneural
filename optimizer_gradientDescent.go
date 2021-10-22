package goneural


type GradientDescentOptimizer struct{
	LearningRate float64
}

type GradientDescentLayer interface{
	Layer
	TrainGradientDescent(learningRate float64, layerInputs, deltas *Matrix) *Matrix
}

func (o *GradientDescentOptimizer) FitOne(n *FeedForwardNetwork, X, Y *Matrix){
	YP := make([]*Matrix, len(n.Layers)+1)
	YP[0] = X
	for i := range n.Layers{
		YP[i+1] = n.Layers[i].PropagateValues(YP[i])

	}
	deltas := n.GetLastLayerDeltas(YP[len(YP)-1], Y)
	for invi := range YP{
		i := len(YP) - 1 - invi
		if i > 0 {
			//don't want to do this for inputs
			deltas = n.Layers[i-1].(GradientDescentLayer).TrainGradientDescent(o.LearningRate, YP[i-1], deltas)
		}
	}
}