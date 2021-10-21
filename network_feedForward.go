package goneural

type FeedForwardNetwork struct{
	Inputs int
	Layers []Layer
	Loss LossFunction
}

func NewFeedForwardNetwork(inputs int, loss LossFunction) *FeedForwardNetwork{
	return &FeedForwardNetwork{
		Inputs: inputs,
		Layers: make([]Layer, 0),
		Loss: loss,
	}
}

func (n *FeedForwardNetwork) AddLayer(layer Layer){
	if len(n.Layers) == 0{
		if n.Inputs != layer.GetNumInputs(){
			panic("Cannot add that layer, it does not have the correct number of inputs")
		}
	} else{
		if n.Layers[len(n.Layers)-1].GetNumOutputs() != layer.GetNumInputs(){
			panic("Cannot add that layer, it does not have the correct number of inputs")
		}
	}
	n.Layers = append(n.Layers, layer)
}

func (n *FeedForwardNetwork) Predict(X []float64) []float64{
	CurrentX := X
	for i := range n.Layers{
		CurrentX = n.Layers[i].PropagateValues(CurrentX)
	}
	return CurrentX
}

func (n *FeedForwardNetwork) GetLastLayerDeltas(pred, expec []float64) []float64{
	deltas := GetLayerLossDiffs(n.Loss, expec, pred)
	for i := range deltas{
		deltas[i] = deltas[i] * n.Layers[len(n.Layers)-1].GetActivation().Diff(pred[i])
	}
	return deltas
}

func (n *FeedForwardNetwork) FitOne(learningRate float64, X, Y []float64) float64{
	YP := make([][]float64, len(n.Layers)+1)
	YP[0] = X
	for i := range n.Layers{
		YP[i+1] = n.Layers[i].PropagateValues(YP[i])

	}
	deltas := n.GetLastLayerDeltas(YP[len(YP)-1], Y)
	for invi := range YP{
		i := len(YP) - 1 - invi
		if i > 0 {
			//dont want to do this for inputs
			deltas = n.Layers[i-1].(GradientDescentLayer).TrainGradientDescent(learningRate, YP[i-1], deltas)
		}
	}
	return GetLayerTotLoss(n.Loss, Y, YP[len(YP)-1])
}