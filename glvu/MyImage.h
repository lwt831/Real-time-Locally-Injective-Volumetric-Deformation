#pragma once

#include <cassert>
#include <vector>
#include <string>
#include <sys/stat.h>
#include "matlab_utils.h"


class MyImage
{
private:
	std::vector<BYTE> pixels;
    int w, h, comp;

public:
    MyImage():w(0),h(0),comp(0) {}
    ~MyImage() { }

	/*MyImage(const std::string &filename, int ncomp=4):w(0), h(0),comp(0)
	{
        // not implemented
    }*/

	MyImage(const std::string &filename) :w(0), h(0), comp(0)		//read image
	{
		const std::string tempname("mytempval4c");
		matlabEval(tempname + std::string("= uint8(imread('") + filename + std::string("'));"), false);
		matlabEval((tempname + std::string("= permute(") + tempname + std::string(",[3,1,2]);")), false);
		mxArray *m = getMatEngine().getVariable(tempname);
		if (!m) {
			printf("read image failed!\n");
			return;
		}
		const mwSize* ImageSize = mxGetDimensions(m);
		comp = ImageSize[0];
		w = ImageSize[1];
		h = ImageSize[2];
		const BYTE *data = (BYTE*)mxGetData(m);
		pixels = std::vector<BYTE>(data, data + comp * w*h);
		mxDestroyArray(m);
		getMatEngine().eval("clear " + tempname + ";");
	}


    MyImage(BYTE* data, int ww, int hh, int pitch, int ncomp = 3) :w(ww), h(hh), comp(ncomp)
    {
        assert(pitch >= ww * 3);
		if (pitch == w*comp) pixels = std::vector<BYTE>(data, data + pitch*h);
		else {
			pixels.resize(w*comp*h);
			for (int i = 0; i < h; i++) std::copy_n(data + pitch*i, pitch, pixels.data() + i*w*comp);
		}
	}

	void img_malloc(int ww, int hh, int ncomp = 3)
	{
		w = ww;
		h = hh;
		comp = ncomp;
		pixels.resize(w * h * comp);
	}

    static int alignment() { return 1; }  // OpenGL only supports 1,2,4,8, do not use 8, it is buggy

    inline bool empty() const { return pixels.empty(); }

	inline BYTE* data() { return pixels.data(); }
	inline const BYTE* data() const { return pixels.data(); }
	inline int width() const { return w; }
	inline int height() const { return h; }
	inline int dim() const { return comp; }
	inline int pitch() const { return w*comp; }


    MyImage resizeCanvas(int ww, int hh)
    {
		std::vector<BYTE> data(ww*comp*hh, 255);
		for (int i = 0; i < h; i++)
			std::copy_n(pixels.data() + i*w*comp, w*comp, data.data() + i*ww*comp);
 
		return MyImage(data.data(), ww, hh, ww*comp, comp);   
	}


    inline void write(const std::string &filename, bool vflip=true) const {
        // not implemented
		mxArray* m;
		mwSize dims[3] = {comp, w, h};
		m = mxCreateNumericArray(3, dims, mxUINT8_CLASS, mxREAL);
		UINT8* mdat = (UINT8 *)mxGetData(m);
		std::memcpy(mdat, pixels.data(), comp * w * h * sizeof(UINT8));
		const std::string tempname("mytempval4c");
		getMatEngine().putVariable(tempname, m);
		matlabEval((tempname + std::string("= permute(") + tempname + std::string(",[3,2,1]);")), false);
		matlabEval(std::string("imwrite(") + tempname + std::string(",'") + filename + std::string("');"));
		mxDestroyArray(m);
		matlabEval("clear " + tempname + ";");
		
	}

	inline std::vector<BYTE> bits(int align=1) const
	{
        const int pitch = (w * comp + align - 1) / align*align;

		std::vector<BYTE> data(pitch*h);
		for(int i=0; i<h; i++)
			std::copy_n(pixels.data()+i*w*comp, w*comp, data.data()+i*pitch);

		return data;
	}
};

