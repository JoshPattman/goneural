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

func (n *FeedForwardNetwork) Predict(X *Matrix) *Matrix{
	CurrentX := X
	for i := range n.Layers{
		CurrentX = n.Layers[i].PropagateValues(CurrentX)
	}
	return CurrentX.Copy()
}

func (n *FeedForwardNetwork) PredictAll(Xs []*Matrix) []*Matrix {
	Ys := make([]*Matrix, len(Xs))
	for i := range Ys{
		Ys[i] = n.Predict(Xs[i])
	}
	return Ys
}

func (n *FeedForwardNetwork) GetLastLayerDeltas(pred, expec *Matrix) *Matrix{
	deltas := GetLayerLossDiffs(n.Loss, expec, pred)
	for i := 0; i < deltas.Shape[0]; i++{
		deltas.SetValue1D(i, deltas.GetValue1D(i) * n.Layers[len(n.Layers)-1].GetActivation().Diff(pred.GetValue1D(i)))
	}
	return deltas
}

