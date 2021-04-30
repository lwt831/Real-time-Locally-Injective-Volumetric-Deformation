#pragma once

#include "deformer.h"
#include "matlab_utils.h"
#include <thread>
#include <chrono>
#include <sstream>  
#include <iostream>  

extern bool showATB;
extern int viewport[4];



extern std::vector<std::string> method_names;
extern int method;
extern double w_smooth;

void display();
void meshDeform();
//void loadP2PConstraints();
using MatX3f = Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>;


inline std::string catStr(const std::vector<std::string> &names)
{
    std::string str;
    for (int i = 0; i < names.size(); i++) {
        str += names[i];
        if (i < names.size() - 1) str += ", ";
    }
    return str;
}

struct MatlabDeformer : public Deformer
{	
	int nIteration = 1;

    MyMesh &M;

    MatlabDeformer(MatlabDeformer&) = delete;

    MatlabDeformer(MyMesh &m) :M(m){

        using deformerptr = MatlabDeformer*;

        //////////////////////////////////////////////////////////////////////////
		method_names = matlab2strings("Deformer_Method");
        std::string default_method = matlab2string("Default_Deformer_Method");
        for (int i = 0; i < method_names.size(); i++) if (default_method == method_names[i]) method = i;
		createDeformer();
		for (int i = 0; i < method_names.size(); i++)
			matlabEval(method_names[i] + std::string(".set_p2p_weight(p2p_weight);"), false);
		
    }

    ~MatlabDeformer(){
        TwBar *bar = TwGetBarByName("Deformer");
        if (bar)    TwDeleteBar(bar); 
    }

   // virtual std::string name(){ return "P2PHarmonic"; }
	virtual void createDeformer()
	{
		matlabEval("createDeformer", false);
	}




    void deformResultFromMaltab(std::string resVarName1, std::string resVarName2)
    {
        using namespace Eigen;
		Matrix<float, Dynamic, Dynamic, RowMajor> y;
		Matrix<float, Dynamic, Dynamic, RowMajor> N;
        matlab2eigen(resVarName1,y,true);
		matlab2eigen(resVarName2, N, true);
		if (y.cols() > 3)  y = y.leftCols(3);
		if (y.rows() == 0 || y.cols() != 3) return;

		if (M.isSmooth)
		{
			MatX3f N;
			matlab2eigen("single(normY)", N, true);
			M.upload(y.data(), y.rows(), N.data(), nullptr, 0, nullptr);
		}
		else
		{			
			M.upload(y.data(), y.rows(), nullptr, 0, nullptr);
			
		}
		
    }

    virtual void deform()
    {
		using namespace Eigen;
		std::vector<int> P2PVtxIds = M.getConstrainVertexIds();
		std::vector<float> p2pDsts = M.getConstrainVertexCoords();
		eigen2matlab("view_P2PVtxIds", (Map<Array<int, Dynamic, 1>>(P2PVtxIds.data(), P2PVtxIds.size()) + 1).matrix().cast<double>());
		eigen2matlab("p2pDsts", Map<MatX3f>(p2pDsts.data(), P2PVtxIds.size(), 3).cast<double>());

		scalar2matlab("nIteration", nIteration);

		matlabEval(std::string("[viewY , Deformation_Converged] = ") + method_names[method] + std::string(".run(nIteration, view_P2PVtxIds, p2pDsts); "), false);
		if (M.isSmooth)
		{
			matlabEval("if(size(viewY,1) ~= size(viewX,1))\
				normY = vertexNormal(triangulation(t, viewY));\
			viewY(end + 1:end + numSeamVertex, : ) = viewY(seamVertexIndex, :);\
			normY(end + 1:end + numSeamVertex, : ) = normY(seamVertexIndex, :);\
			end", false);
		}
		else{
			matlabEval("if(size(viewY,1) ~= size(viewX,1))\
		viewY(end + 1:end + numSeamVertex, : ) = viewY(seamVertexIndex, :);\
		end", false);
		}
		//matlabEval("updateNormal", false);
        deformResultFromMaltab("single(viewY)", "single(normY)");
		if (M.vizVtxDistortion)
		{
			
			matlabEval(std::string("distortion = ") + method_names[method] + std::string(".computeVertexDistortion;"), false);
			matlabEval(std::string("distortion(end+1:end+numSeamVertex) = distortion(seamVertexIndex);"), false);

			auto distortion = matlab2vector<float>("single(distortion)", true);
			M.gpuVertexData.uploadData(distortion.data(), M.nVertex, false);
		}
    }

    virtual bool converged() {
        return !getMatEngine().hasVar("Deformation_Converged")  ||  matlab2scalar("Deformation_Converged") > 0;
    }

    virtual void resetDeform() {
		matlabEval(method_names[method] + std::string(".Reset;"), false);
		matlabEval("viewY = viewX;", false);
		matlabEval("normY = normX;", false);
		matlabEval("Deformation_Converged = 0", false);
		printf("====================================\n");
		printf("		reset mesh			\n");
		deformResultFromMaltab("single(viewY)", "single(normX)");
    }
    virtual void getResult() {}
    //virtual void saveData()   { matlabEval("p2p_harmonic_savedata;"); }
};
