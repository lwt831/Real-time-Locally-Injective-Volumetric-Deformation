#pragma once
#include"AntTweakBar.h"
#include "vaomesh.h"

struct Deformer
{
    bool needIteration = true;
    virtual std::string name(){ return "UNKNOWN"; }
    virtual void preprocess(bool) {}
    virtual void deform() = 0;
	virtual void createDeformer() {};
    virtual bool converged() { return false; }
    virtual void resetDeform(){}
    virtual void getResult(){}
    virtual void saveData(){}
};
