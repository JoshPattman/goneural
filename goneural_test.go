package goneural

import "testing"

func TestFeedForward(t *testing.T) {
	l, err := NewFeedForwardLayer(5, 3, "relu")
	checkErr(err, t)
	vals, err := l.Calculate([]float64{})
	checkErr(err, t)
	if len(vals) != 3{
		t.Error("Layer did not give expected number of outputs")
	}

}

func checkErr(err error, t *testing.T){
	if err != nil{
		t.Error(err.Error())
	}
}