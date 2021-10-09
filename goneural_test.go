package goneural

import (
	"fmt"
	"testing"
)

func TestFP(t *testing.T) {
	n := createNetwork()
	fmt.Println(n.Calculate([]float64{0, 1, -2}))
	fmt.Println(n.ToJSON())
}


func createNetwork()*Network{
	l1 := NewFeedForwardLayer(3, 5, "linear")
	l2 := NewFeedForwardLayerAfter(l1, 5, "leaky_relu")
	l3 := NewFeedForwardLayerAfter(l2, 1, "sigmoid")
	n := NewNetwork([]Layer{
		l1,
		l2,
		l3,
	})
	return n
}
