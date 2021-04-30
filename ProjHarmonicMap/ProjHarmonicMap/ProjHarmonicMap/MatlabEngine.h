#pragma once

//#define MEX_DOUBLE_HANDLE	
#include <engine.h>
#include <cstdint>
#include <cstdarg>
#include<string>
#include <vector>

#define ENABLE_MATLAB

inline void ensure(bool cond, const char* msg, ...)
{
	if (!cond) { va_list args; va_start(args, msg); vfprintf(stderr, (msg + std::string("\n")).c_str(), args); va_end(args); }
}

#define ensureTypeMatch(R, m, othertype) ensure(MatlabNum<R>::id == mxGetClassID(m), "Matlab type does not match "##othertype)

class MatlabEngine
{
public:
	bool consoleOutput;

	MatlabEngine() :eng(nullptr), consoleOutput(true) { }
	virtual ~MatlabEngine() { if (eng) close(); }

	// Run inside matlab: enableservice('AutomationServer', true)
	bool connect(const std::string& dir, bool closeAll = false);

	bool connected() const { return eng != nullptr; }
	void setEnable(bool v) { if (v) connect(""); else close(); }

	void eval(const std::string& cmd, bool PrintInfo = true);

	void close();

	void hold_on() { eval("hold on;"); }

	void hold_off() { eval("hold off;"); }

	const char* output_buffer() { return (*engBuffer) ? engBuffer : nullptr; }

	bool hasVar(const std::string& name)
	{
		ensure(connected(), "Not connected to Matlab!");
		mxArray* m = engGetVariable(eng, name.c_str());
		bool r = (m != nullptr);
		mxDestroyArray(m);
		return r;
	}

	mxArray* getVariable(const std::string& name)
	{
		ensure(connected(), "Not connected to Matlab!");
		mxArray* m = engGetVariable(eng, name.c_str());
		ensure(m != nullptr, "Matlab doesn't have a variable: %s\n", name.c_str());

		return m;
	}
	mxArray* getVariable(const char* name)
	{
		ensure(connected(), "Not connected to Matlab!");
		mxArray* m = engGetVariable(eng, name);
		ensure(m != nullptr, "Matlab doesn't have a variable: %s\n", name);

		return m;
	}

	int putVariable(const std::string& name, const mxArray* m)
	{
		ensure(connected(), "Not connected to Matlab!");	return engPutVariable(eng, name.c_str(), m);
	}


private:
	Engine* eng; // Matlab engine
	static const int lenEngBuffer = 1000000;
	char engBuffer[lenEngBuffer]; // engine buffer for outputting strings
};


MatlabEngine& getMatEngine();
// inline void matlabEval(const char* cmd) { getMatEngine().eval(cmd); }
inline void matlabEval(const std::string& cmd, bool PrintInfo = true) { getMatEngine().eval(cmd.c_str(), PrintInfo); }

inline bool matEngineConnected() { return getMatEngine().connected(); }



