package goneural

import "fmt"

type SerialTrainer struct{
	Opt Optimizer
}

func (s *SerialTrainer) Fit(n *FeedForwardNetwork, Xs, Ys [][]float64, epochs, printRate int){
	for e := range make([]int, epochs) {
		if e % printRate == 0{
			loss := 0.0
			for i := range Xs{
				s.Opt.FitOne(n, Xs[i], Ys[i])
				loss += GetLayerTotLoss(n.Loss, Ys[i], n.Predict(Xs[i]))
			}
			fmt.Println("Epoch: ", e, ", Loss (mse): ", loss/float64(len(Xs)))
		} else {
			for i := range Xs{
				s.Opt.FitOne(n, Xs[i], Ys[i])
			}
		}
	}
}