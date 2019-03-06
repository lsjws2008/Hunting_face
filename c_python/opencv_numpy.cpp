#ifdef _DEBUG
#undef _DEBUG
#include <python3.6m/Python.h>
#define _DEBUG
#else
#include <python3.6m/Python.h>
#endif
#include <iostream>
#include <stdlib.h>
#include <numpy/arrayobject.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
//g++ -I/usr/include/python3.6m/ opencv_numpy.cpp -lpython3.6m -L/usr/include/opencv -lopencv_core -lopencv_imgcodecs -lopencv_imgproc
//PYTHONPATH=. ./call_function pyfunction multiply 2 3
//PYTHONPATH=. ./a.out  test_api imgs_to_out ~/PycharmProjects/Hunting_face/test_dir/
using namespace std;
int
main(int argc, char *argv[])
{
    setenv("PYTHONPATH",".",1);
    PyObject *pName = nullptr, *pModule = nullptr, *pFunc = nullptr;
    PyObject *pArgs = nullptr, *pReturn = nullptr, *pString = nullptr;
    if (argc < 3) {
        fprintf(stderr, "Usage: call pythonfile funcname [args]\n");
        return EXIT_FAILURE;
    }
    
    wchar_t *wcsProgram = Py_DecodeLocale(argv[0], NULL);
    
    Py_SetProgramName(wcsProgram);
    
    Py_Initialize();
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append(\".\")");
    
    pName = PyUnicode_DecodeFSDefault(argv[1]);
    if (PyErr_Occurred()) {
        std::cerr << "pName decoding failed." << std::endl;
        return EXIT_FAILURE;
    }
    pModule = PyImport_Import(pName);
    
    Py_DECREF(pName);
    
    if (pModule != NULL) {
        pFunc = PyObject_GetAttrString(pModule, argv[2]);
	
	if (pFunc && PyCallable_Check(pFunc)) {
            //cv::Mat img = cv::imread(argv[4], cv::IMREAD_COLOR), img_rgb;
	    
            //cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);
            // Build the array in C++
            //const int ND = 2;
            //npy_intp dims[2]{ img_rgb.rows, img_rgb.cols * img_rgb.channels() };
            // Convert it to a NumPy array.
            //import_array();
            //PyObject *pArray = PyArray_SimpleNewFromData(
            //    ND, dims, NPY_UBYTE, reinterpret_cast<void*>(img_rgb.data));
            //if (!pArray) {
            //    std::cerr << "PyArray_SimpleNewFromData failed." << std::endl;
            //    return EXIT_FAILURE;
            //}
            //PyObject *pValue = PyLong_FromLong(img_rgb.channels());
            pString = PyUnicode_FromString(argv[3]);
	    
            pReturn = PyObject_CallFunctionObjArgs(pFunc, pString, argv[4], NULL);
            if (pReturn != NULL) {
                fprintf(stdout, "Result of call: %ld\n", PyLong_AsLong(pReturn));
                Py_DECREF(pReturn);
            }
            else {
                Py_DECREF(pFunc);
                Py_DECREF(pModule);
                PyErr_Print();
                fprintf(stderr, "Call failed\n");
                return EXIT_FAILURE;
            }
        }
    
        else {
            if (PyErr_Occurred())
                PyErr_Print();
            fprintf(stderr, "Cannot find function \"%s\"\n", argv[2]);
        }
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
    }
    else {
        PyErr_Print();
        fprintf(stderr, "Failed to load \"%s\"\n", argv[1]);
        return EXIT_FAILURE;
    }
    
    Py_Finalize();
    return EXIT_SUCCESS;
}
