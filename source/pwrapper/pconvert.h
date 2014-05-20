/******************************************************************************
 *
 * MantaFlow fluid solver framework
 * Copyright 2011 Tobias Pfaff, Nils Thuerey 
 *
 * This program is free software, distributed under the terms of the
 * GNU General Public License (GPL) 
 * http://www.gnu.org/licenses
 *
 * Python argument wrappers and conversion tools
 *
 ******************************************************************************/

// -----------------------------------------------------------------
// NOTE:
// Do not include this file in user code, include "manta.h" instead
// -----------------------------------------------------------------

#ifdef _MANTA_H
#ifndef _PCONVERT_H
#define _PCONVERT_H

#include <string>
#include <map>
#include <vector>

namespace Manta { 
template<class T> class Grid; 


//! Locks the given PbClass Arguments until ArgLocker goes out of scope
struct ArgLocker {    
    void add(PbClass* p);
    ~ArgLocker();
    std::vector<PbClass*> locks;
};

PyObject* getPyNone();

// for PbClass-derived classes
template<class T> T* fromPyPtr(PyObject* obj) { 
    if (PbClass::isNullRef(obj)) 
        return 0; 
    PbClass* pbo = Pb::objFromPy(obj); 
    const std::string& type = Namify<T>::S;
    if (!pbo || !(pbo->canConvertTo(type))) 
        throw Error("can't convert argument to " + type); 
    return (T*)(pbo); 
}

template<class T> T fromPy(PyObject* obj) {
    return *fromPyPtr<typename remove_pointers<T>::type>(obj);
}

template<class T> PyObject* toPy(const T& v) { 
    if (v.getPyObject()) 
        return v.getPyObject(); 
    T* co = new T (v); 
    const std::string& type = Namify<typename remove_pointers<T>::type>::S;
    return Pb::copyObject(co,type); 
}
template<class T> bool isPy(PyObject* obj) {
    if (PbClass::isNullRef(obj)) 
        return false; 
    PbClass* pbo = Pb::objFromPy(obj); 
    const std::string& type = Namify<typename remove_pointers<T>::type>::S;
    return pbo && pbo->canConvertTo(type);
}

// builtin types
template<> float fromPy<float>(PyObject* obj);
template<> double fromPy<double>(PyObject* obj);
template<> int fromPy<int>(PyObject *obj);
template<> PyObject* fromPy<PyObject*>(PyObject *obj);
template<> std::string fromPy<std::string>(PyObject *obj);
template<> const char* fromPy<const char*>(PyObject *obj);
template<> bool fromPy<bool>(PyObject *obj);
template<> Vec3 fromPy<Vec3>(PyObject* obj);
template<> Vec3i fromPy<Vec3i>(PyObject* obj);
template<> PbType fromPy<PbType>(PyObject* obj);

template<> PyObject* toPy<int>( const int& v);
template<> PyObject* toPy<std::string>( const std::string& val);
template<> PyObject* toPy<float>( const float& v);
template<> PyObject* toPy<double>( const double& v);
template<> PyObject* toPy<bool>( const bool& v);
template<> PyObject* toPy<Vec3i>( const Vec3i& v);
template<> PyObject* toPy<Vec3>( const Vec3& v);
typedef PbClass* PbClass_Ptr;
template<> PyObject* toPy<PbClass*>( const PbClass_Ptr & obj);

template<> bool isPy<float>(PyObject* obj);
template<> bool isPy<double>(PyObject* obj);
template<> bool isPy<int>(PyObject *obj);
template<> bool isPy<PyObject*>(PyObject *obj);
template<> bool isPy<std::string>(PyObject *obj);
template<> bool isPy<const char*>(PyObject *obj);
template<> bool isPy<bool>(PyObject *obj);
template<> bool isPy<Vec3>(PyObject* obj);
template<> bool isPy<Vec3i>(PyObject* obj);
template<> bool isPy<PbType>(PyObject* obj);

//! Encapsulation of python arguments
class PbArgs {
public:
    PbArgs(PyObject *linargs = NULL, PyObject* dict = NULL);
    void setup(PyObject *linargs = NULL, PyObject* dict = NULL);
    
    void check();
    FluidSolver* obtainParent();
    
    inline int numLinArgs() { return mLinData.size(); }
    
    inline bool has(const std::string& key) {
        return getItem(key, false) != NULL;
    }
    
    inline PyObject* linArgs() { return mLinArgs; }
    inline PyObject* kwds() { return mKwds; }
    
    template<class T> inline void add(const std::string& key, T arg) {
        DataElement el = { toPy(arg), false };
        mData[key] = el;
    }
    template<class T> inline T get(const std::string& key, int number=-1, ArgLocker *lk=NULL) {
        PyObject* o = getItem(key, false, lk);
        if (o) return fromPy<T>(o);
        o = getItem(number, false, lk);
        if (o) return fromPy<T>(o);
        errMsg ("Argument '" + key + "' is not defined.");        
    }
    template<class T> inline T getOpt(const std::string& key, int number, T defarg, ArgLocker *lk=NULL) { 
        PyObject* o = getItem(key, false, lk);
        if (o) return fromPy<T>(o);
        if (number >= 0) o = getItem(key, false);
        return (o) ? fromPy<T>(o) : defarg;
    }
    template<class T> inline T* getPtrOpt(const std::string& key, int number, T* defarg, ArgLocker *lk=NULL) {
        PyObject* o = getItem(key, false, lk);
        if (o) return fromPyPtr<T>(o);
        if (number >= 0) o = getItem(number, false);
        return o ? fromPyPtr<T>(o) : defarg;
    }
    template<class T> inline T* getPtr(const std::string& key, int number = -1, ArgLocker *lk=NULL) {
        PyObject* o = getItem(key, false, lk);
        if (o) return fromPyPtr<T>(o);
        o = getItem(number, false);
        if(o) return fromPyPtr<T>(o);
        errMsg ("Argument '" + key + "' is not defined.");
    }


    // automatic template type deduction
    template<class T> bool typeCheck(int num, const std::string& name) {
        PyObject* o = getItem(name, false, 0);
        if (!o) 
            o = getItem(num, false, 0);
        return o ? isPy<T>(o) : false;
    }
    
    PbArgs& operator=(const PbArgs& a); // dummy
    void copy(PbArgs& a);
    void clear();
    
    static PbArgs EMPTY;
    
protected:
    PyObject* getItem(const std::string& key, bool strict, ArgLocker* lk = NULL);
    PyObject* getItem(size_t number, bool strict, ArgLocker* lk = NULL);    
    
    struct DataElement {
        PyObject *obj;
        bool visited;
    };
    std::map<std::string, DataElement> mData;
    std::vector<DataElement> mLinData;
    PyObject* mLinArgs, *mKwds;    
};


} // namespace
#endif
#endif
