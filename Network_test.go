package goneural

import (
	"fmt"
	"testing"
)

func TestFeedForward(t *testing.T){
	lastLayer := []float64{1, 0, 1, 0, 1}
	l := NewFeedForwardLayer(len(lastLayer), 3, "linear")
	fmt.Println(l.Calculate(lastLayer))

	js := l.ToJSON()
	fmt.Println(js)
	l2 := NewFeedForwardLayerFromJSON(js)
	if l2.ToJSON() != l.ToJSON(){
		t.Error("JSON Serialisation not working")
	}
}

func TestNetwork(t *testing.T){
	inputs := []float64{1, 0, 1, 0, 1}
	l1 := NewFeedForwardLayer(len(inputs), 3, "linear")
	l2 := NewFeedForwardLayerAfter(l1, 1, "linear")
	n := NewNetwork([]Layer{l1, l2})
	fmt.Println(n.Calculate(inputs))
	js := n.ToJSON()
	fmt.Println(js)
	n2 := Network{}
	n2.LoadJSON(js)
	fmt.Println(n2.ToJSON())
	//t.Error("Not Implemented Yet")
}

