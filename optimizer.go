package goneural

type Optimizer interface{
	FitOne(n *FeedForwardNetwork, X, Y []float64)
}