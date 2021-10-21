package goneural

import "fmt"

type SerialTrainer struct{
	Opt Optimizer
}

func (s *SerialTrainer) Fit(n *FeedForwardNetwork, Xs, Ys [][]float64, epochs, printRate int){
	for e := range make([]int, epochs) {
		loss := 0.0
		for i := range Xs{
			loss += s.Opt.FitOne(n, Xs[i], Ys[i])
		}
		if e % printRate == 0{
			fmt.Println("Epoch: ", e, ", Loss (mse): ", loss/float64(len(Xs)))
		}
	}
}