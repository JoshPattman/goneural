package goneural

import "strconv"

// BasicShapeError : An error that just stores a ing that is called by a layer to say that the shape is wrong
type BasicShapeError struct{
	s string
}

func (se BasicShapeError) Error()string{
	return se.s
}

// ShapeError : What a BasicShapeError is converted to by a network that adds the layers index
type ShapeError struct{
	s string
	// The layer number
	l int
}

func (se ShapeError) Error()string{
	return "Shape Error (layer " + strconv.Itoa(se.l) +"): "+se.s
}

// ActivationNotFoundError : Called when a string activation function is not found
type ActivationNotFoundError struct{
	activation string
}

func (a ActivationNotFoundError) Error()string{
	return "Activation Not Found Error: Could not find activation '"+a.activation+"'"
}
