/******************************************************************************
 *
 * MantaFlow fluid solver framework
 * Copyright 2017-2018 Steffen Wiewel, Moritz Becher, Rachel Chu
 *
 * This program is free software, distributed under the terms of the
 * Apache License, Version 2.0
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Convert mantaflow grids to/from numpy arrays
 *
 * Modifications copyright (c) 2023 zarsten
 * 
 * modifications: change long to Py_intptr_t in all dims related code,
 * in order to fix size miss match on msvc (long = 4 bytes, Py_intptr_t = 8 bytes in x64 builds)
 * 
 * Change DataType to accept all types of signed integers as N_INT, instead of just NPY_INT,
 * which is not guaranteed to be available on all platforms, when using cnpy for storing and loading npy files.
 * 
 ******************************************************************************/

#include "manta.h"
#include "pythonInclude.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

namespace Manta {

#if PY_VERSION_HEX < 0x03000000
PyMODINIT_FUNC initNumpy() { import_array(); }
#endif

// ------------------------------------------------------------------------
// Class Functions
// ------------------------------------------------------------------------
PyArrayContainer::PyArrayContainer(void *_pParentPyArray) :
	pParentPyArray(_pParentPyArray)
{
	ExtractData(pParentPyArray);
}
// ------------------------------------------------------------------------
PyArrayContainer::PyArrayContainer(const PyArrayContainer &_Other) :
	pParentPyArray(_Other.pParentPyArray)
{
	ExtractData(pParentPyArray);
	Py_INCREF(pParentPyArray);
}
// ------------------------------------------------------------------------
PyArrayContainer::~PyArrayContainer()
{
	Py_DECREF(pParentPyArray);
}
// ------------------------------------------------------------------------
PyArrayContainer &
PyArrayContainer::operator=(const PyArrayContainer &_Other)
{
	if(this != &_Other) {
		// DecRef the existing resource
		Py_DECREF(pParentPyArray);

		// Relink new data
		pParentPyArray = _Other.pParentPyArray;
		ExtractData(pParentPyArray);
		Py_INCREF(pParentPyArray);
	}
	return *this;
}
// ------------------------------------------------------------------------
void
PyArrayContainer::ExtractData(void *_pParentPyArray)
{
	PyArrayObject *pParent = reinterpret_cast<PyArrayObject *>(pParentPyArray);

	int numDims = PyArray_NDIM(pParent);
	Py_intptr_t* pDims = (Py_intptr_t*)PyArray_DIMS(pParent);

	pData 		= PyArray_DATA(pParent);
	TotalSize 	= PyArray_SIZE(pParent);
	Dims 		= std::vector<Py_intptr_t>(&pDims[0], &pDims[numDims]);
	
	int iDataType = PyArray_TYPE(pParent);
	switch(iDataType) {
	case NPY_FLOAT:
		DataType = N_FLOAT;
		break;
	case NPY_DOUBLE:
		DataType = N_DOUBLE;
		break;
	default:
		if (PyArray_ISNUMBER(pParent) && PyArray_ISSIGNED(pParent))
			DataType = N_INT;
		else
			errMsg("unknown type of Numpy array");
		break;

		
	}
}

// ------------------------------------------------------------------------
// Conversion Functions
// ------------------------------------------------------------------------

template<>
PyArrayContainer
fromPy<PyArrayContainer>(PyObject *obj)
{
	if(PyArray_API == NULL) {
		// python 3 uses the return value
#		if PY_VERSION_HEX >= 0x03000000
		import_array();
#		else
		initNumpy();
#		endif
	}

	if(!PyArray_Check(obj)) {
		errMsg("argument is not an numpy array");
	}

	PyArrayObject *obj_p = reinterpret_cast<PyArrayObject *>(PyArray_CheckFromAny(obj, NULL, 0, 0, /*NPY_ARRAY_ENSURECOPY*/ NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ENSUREARRAY | NPY_ARRAY_NOTSWAPPED, NULL));
	PyArrayContainer container = PyArrayContainer(obj_p);

	return container;
}

// template<> PyArrayContainer* fromPyPtr<PyArrayContainer>(PyObject* obj, std::vector<void*>* tmp)
// {
// 	if (!tmp) throw Error("dynamic de-ref not supported for this type");
// 	void* ptr = malloc(sizeof(PyArrayContainer));
// 	tmp->push_back(ptr);

// 	*((PyArrayContainer*) ptr) = fromPy<PyArrayContainer>(obj);
// 	return (PyArrayContainer*) ptr;
// }
}
