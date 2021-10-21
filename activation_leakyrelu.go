package goneural

type LeakyReLUActivation struct{
	SubZeroGradient float64
}

func (a LeakyReLUActivation) Calc(x float64) float64{
	if x > 0{
		return x
	} else{
		return x*a.SubZeroGradient
	}
}

func (a LeakyReLUActivation) Diff(x float64)float64{
	if x > 0{
		return 1
	} else{
		return a.SubZeroGradient
	}
}