package goneural

type Activation interface{
	Calc(x float64) float64
	Diff(x float64) float64
}