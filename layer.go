package goneural

type Layer interface{
	// Calculate the layer by taking the values of the prev layer, doing some maths with our weights, then returning a list of the values of this layer
	Calculate(previousValues []float64) ([]float64, error)
	// GetNumNeurons gets the number of neurons (outputs) in this layer
	GetNumNeurons() int

	GetLayerNameID() string

	ToJSON() string
	LoadJSON(string)
}


