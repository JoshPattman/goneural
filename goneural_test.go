package goneural

import "testing"

func TestFP(t *testing.T) {

}

func createNetwork()*Network{
	l1 := NewFeedForwardLayer(3, 5, "linear")
	l2 := NewFeedForwardLayerAfter(l1, 5, "linear")
	l3 := NewFeedForwardLayerAfter(l2, 1, "linear")
	n := NewNetwork([]Layer{
		l1,
		l2,
		l3,
	})
	return n
}