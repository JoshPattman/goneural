package data

type ParseFunction func(RawRow) Row

type RawRow []string
type Row []float64

func (s RawRow) Convert(parseFunction ParseFunction) Row {
	return parseFunction(s)
}