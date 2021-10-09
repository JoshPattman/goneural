package goneural

import (
	"encoding/json"
	"strconv"
)

///////////////////////////////////// TYPES /////////////////////////////////////

type Layer interface{
	// Calculate the layer by taking the values of the prev layer, doing some maths with our weights, then returning a list of the values of this layer
	Calculate(previousValues []float64) ([]float64, error)
	// GetNumNeurons gets the number of neurons (outputs) in this layer
	GetNumNeurons() int

	GetLayerNameID() string

	ToJSON() string
	LoadJSON(string)
}


///////////////////////////////////// FEED FORWARD LAYER /////////////////////////////////////

// FeedForwardLayer : The standard feed forward neural network layer that multiplies weights values then adds biases
type FeedForwardLayer struct {
	PrevNeurons int       `json:"prev_neurons"`
	ThisNeurons int       `json:"this_neurons"`
	Weights     []float64 `json:"weights"`
	Biases      []float64  `json:"biases"`
	ActivationString string `json:"activation"`
	activation       ActivationFunction
}

// GetLayerNameID : Gets the nameID for this type of layer (eg feed_forward)
func (layer *FeedForwardLayer) GetLayerNameID() string {
	return "feed_forward"
}

// Calculate :  Calculate the values of this layer using the values of the previous layer
func (layer *FeedForwardLayer) Calculate(previousValues []float64) ([]float64, error){
	if len(previousValues) != layer.PrevNeurons {
		return []float64{}, BasicShapeError{"Incorrect number of inputs to layer. Got " + strconv.Itoa(len(previousValues)) + ", wanted " + strconv.Itoa(layer.PrevNeurons)}
	}
	values := make([]float64, layer.ThisNeurons)
	for neuron := range values{
		for prevNeuron := range previousValues{
			values[neuron] += previousValues[prevNeuron] * (*layer).getWeight(prevNeuron, neuron)
		}
		values[neuron] = layer.activation(values[neuron])
	}
	return values, nil
}

// GetNumNeurons : Gets the number of neurons (outputs) of this layer
func (layer *FeedForwardLayer) GetNumNeurons() int{
	return layer.ThisNeurons
}

// NewFeedForwardLayer : Creates a new feed forward layer with standard parameters
func NewFeedForwardLayer(prevNeurons int, thisNeurons int, activation string) (*FeedForwardLayer, error){
	if ActivationFunc, ok := activationFunctions[activation]; ok {
		return &FeedForwardLayer{
			PrevNeurons:      prevNeurons,
			ThisNeurons:      thisNeurons,
			Weights:          makeRandomSlice(prevNeurons*thisNeurons, -1, 1),
			Biases:           makeRandomSlice(thisNeurons, -1, 1),
			ActivationString: activation,
			activation:       ActivationFunc,
		}, nil
	}
	return nil, ActivationNotFoundError{activation}
}

// NewFeedForwardLayerAfter : Creates a new feed forward layer, but uses the number of nodes in the previous layer as the number of inputs in this layer
func NewFeedForwardLayerAfter(prevLayer Layer, thisNeurons int, activation string)(*FeedForwardLayer, error){
	return NewFeedForwardLayer(prevLayer.GetNumNeurons(), thisNeurons, activation)
}

// NewFeedForwardLayerFromJSON : Creates a new feed forward layer from a json serialisation
func NewFeedForwardLayerFromJSON(s string)*FeedForwardLayer{
	layer := FeedForwardLayer{}
	layer.LoadJSON(s)
	return &layer
}

// getWeight : Gets the weight of this layer coming from fromNeuron and to toNeuron
func (layer *FeedForwardLayer) getWeight(fromNeuron, toNeuron int) float64{
	return layer.Weights[(fromNeuron*layer.ThisNeurons) + toNeuron]
}

// LoadJSON : Loads the json string into this layer
func (layer *FeedForwardLayer) LoadJSON(s string){
	err := json.Unmarshal([]byte(s), layer)
	if err != nil{
		panic(err)
	}
	if acfunc, ok := activationFunctions[layer.ActivationString]; ok{
		layer.activation = acfunc
	} else{
		panic("Did not recognise activation function " + layer.ActivationString)
	}
}

// ToJSON : Creates a json representation of this layer
func (layer *FeedForwardLayer) ToJSON()string{
	js, err := json.Marshal(*layer)
	if err != nil{
		panic(err)
	}
	return string(js)
}
