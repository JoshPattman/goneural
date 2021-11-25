package goneural

type MinibatchOptimizer struct{
	LearningRate float64
	BatchSize int
}


type MinibatchLayer interface{
	Layer
	TrainMinibatch(learningRate float64, layerInputs, totDeltas *Matrix) *Matrix
}

func (o *MinibatchOptimizer) FitBatch(n *FeedForwardNetwork, X, Y []*Matrix){
	numLayers := len(n.Layers)
	totLayerInputs := make([]*Matrix, numLayers)
	for i := range totLayerInputs{
		totLayerInputs[i] = New1DMatrix(make([]float64, n.Layers[i].GetNumInputs()))
	}
	totalDeltas := New1DMatrix(make([]float64, n.Outputs))
	for sample := range X{
		x := X[sample]
		y := Y[sample]
		// Calc deltas
		yp := make([]*Matrix, numLayers+1)
		yp[0] = x
		totLayerInputs[0].Add(x)
		for i := range n.Layers {
			yp[i+1] = n.Layers[i].PropagateValues(yp[i])
			if i < len(n.Layers)-1 {
				totLayerInputs[i+1].Add(yp[i+1])
			}
		}
		totalDeltas.Add(n.GetLastLayerDeltas(yp[len(yp)-1], y))
	}

	N := 1.0/float64(len(X))
	avDeltas := totalDeltas.Mul(N)
	avLayerInputs := totLayerInputs
	for i := range avLayerInputs{
		avLayerInputs[i].Mul(N)
	}

	for inverseI := 0; inverseI < numLayers; inverseI ++ {
		i := numLayers - 1 - inverseI
		//fmt.Println("Layer", i, ":", n.Layers[i].GetNumOutputs(), avDeltas.Shape, n.Layers[i].GetNumInputs(), avLayerInputs[i].Shape)
		avDeltas = n.Layers[i].(MinibatchLayer).TrainMinibatch(o.LearningRate, avLayerInputs[i], avDeltas)
	}
}

func (o *MinibatchOptimizer)GetBatchSize()int{
	return o.BatchSize
}