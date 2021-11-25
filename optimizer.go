package goneural

type Optimizer interface{
	FitBatch(n *FeedForwardNetwork, X, Y []*Matrix)
	GetBatchSize() int
}
