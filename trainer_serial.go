package goneural

import "fmt"

type SerialTrainer struct{
	Opt Optimizer
	UseRelativeLearningRate bool
}

func (s *SerialTrainer) Fit(n *FeedForwardNetwork, Xs, Ys []*Matrix, epochs, printRate int){
	batchSize := s.Opt.GetBatchSize()
	numXs := ((len(Xs) / batchSize) - 1) * batchSize
	for e := range make([]int, epochs) {
		if e % printRate == 0{
			loss := 0.0
			for i := 0; i < numXs; i += s.Opt.GetBatchSize(){
				s.Opt.FitBatch(n, Xs[i:i+batchSize], Ys[i:i+batchSize])
				loss += GetLayerTotLoss(n.Loss, Ys[i], n.Predict(Xs[i]))
			}
			loss = loss/float64(len(Xs))
			fmt.Println("Epoch: ", e, ", Loss (mse): ", loss)
		} else {
			for i := 0; i < numXs; i += s.Opt.GetBatchSize(){
				s.Opt.FitBatch(n, Xs[i:i+batchSize], Ys[i:i+batchSize])
			}
		}
	}
}