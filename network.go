package goneural

import (
	"encoding/json"
	"strconv"
)

// Network : A feed forward neural network
type Network struct{
	// Layers : A slice of layers. The layers in here are actually *TypeOfLayer, but the reference is the type that implements Layer interface
	Layers []Layer
}

// SerialisedNetwork : A half serialised representation of a network that is ready for full json serialisation
type SerialisedNetwork struct{
	// LayerTypes : The string representation of the different layers types (feed_forward, recursive)
	LayerTypes []string `json:"layer_types"`
	// Layers : The string representation of each layer in the network this is based on
	Layers []string `json:"layers"`
}

// NewNetwork : Creates a new network from a slice of layers
func NewNetwork(Layers []Layer) *Network{
	return &Network{
		Layers: Layers,
	}
}

// ToJSON : Converts the network to the json format for saving
func (net *Network) ToJSON() string{
	layers := make([]string, len(net.Layers))
	layerTypes := make([]string, len(net.Layers))
	for l := range layers{
		layers[l] = net.Layers[l].ToJSON()
		layerTypes[l] = net.Layers[l].GetLayerNameID()
	}
	sn := SerialisedNetwork{
		Layers: layers,
		LayerTypes: layerTypes,
	}
	js, err := json.Marshal(sn)
	if err != nil{
		panic(err)
	}
	return string(js)
}

// LoadJSON : Loads a serialised network in the json format into the network
func (net *Network) LoadJSON(js string){
	sn := SerialisedNetwork{}
	err := json.Unmarshal([]byte(js), &sn)
	if err != nil{
		panic("Failed to unmarshal network")
	}
	net.Layers = make([]Layer, len(sn.Layers))
	for i := range sn.Layers{
		if sn.LayerTypes[i] == (&FeedForwardLayer{}).GetLayerNameID(){
			l := FeedForwardLayer{}
			l.LoadJSON(sn.Layers[i])
			net.Layers[i] = &l
		}
	}
}

// Calculate : Propagate input values through the network and return the result
func (net *Network) Calculate(inputs []float64) ([]float64, error){
	lastLayerValues := inputs
	for l := range net.Layers{
		var err error
		lastLayerValues, err = net.Layers[l].Calculate(lastLayerValues)
		if err != nil{
			switch err.(type){
			case BasicShapeError:
				err = ShapeError{err.Error(), l}
			}
			return []float64{}, err
		}
	}
	return lastLayerValues, nil
}

func (net *Network) Summary() string{
	s := "Network with " + strconv.Itoa(len(net.Layers)) + " layers"
	return s
}
