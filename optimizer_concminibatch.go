package goneural

type ConcMinibatchOptimizer struct{
	LearningRate float64
	BatchSize int
	SamplesPerGR int
}

func (o *ConcMinibatchOptimizer) FitBatch(n *FeedForwardNetwork, X, Y []*Matrix){
	numLayers := len(n.Layers)
	totLayerInputs := make([]*Matrix, numLayers)
	for i := range totLayerInputs{
		totLayerInputs[i] = New1DMatrix(make([]float64, n.Layers[i].GetNumInputs()))
	}
	totalDeltas := New1DMatrix(make([]float64, n.Outputs))
	deltaChan := make(chan *Matrix, len(X))
	inpChan := make(chan []*Matrix, len(X))
	// start goroutines
	lenX := len(X)
	for sample := 0; sample < lenX; sample += o.SamplesPerGR {
		go func() {
			for s := 0; s < o.SamplesPerGR; s++ {
				if s < lenX {
					x := X[s]
					y := Y[s]
					// Calc deltas
					yp := make([]*Matrix, numLayers+1)
					yp[0] = x
					for i := range n.Layers {
						yp[i+1] = n.Layers[i].PropagateValues(yp[i])
					}
					deltaChan <- n.GetLastLayerDeltas(yp[len(yp)-1], y)
					inpChan <- yp
				}
			}
		}()
	}
	//collect info
	for range X{
		totalDeltas.Add(<- deltaChan)
		inps := <- inpChan
		for i := range totLayerInputs{
			totLayerInputs[i].Add(inps[i])
		}
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

func (o *ConcMinibatchOptimizer)GetBatchSize()int{
	return o.BatchSize
}