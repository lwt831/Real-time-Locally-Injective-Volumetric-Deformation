#pragma once
#include <algorithm>
#include <vector>
#include <array>
#include<iostream>
#include <random> 

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include "glprogram.h"
#include "glarray.h"
#include "matlab_utils.h"

class GLGeoFbo
{
public:
	GLuint gPosition, gNormalAndShadow, gColor, rboDepth, gBuffer;
	GLGeoFbo() :gPosition(0), gNormalAndShadow(0), gColor(0), rboDepth(0), gBuffer(0){};
	~GLGeoFbo() { deleteGeoFbo();};
	void createGeoFbo(int WIDTH,int HEIGHT)
	{
		glGenFramebuffers(1, &gBuffer);
		glBindFramebuffer(GL_FRAMEBUFFER, gBuffer);
		glGenTextures(1, &gPosition);
		glBindTexture(GL_TEXTURE_2D, gPosition);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, WIDTH, HEIGHT, 0, GL_RGB, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gPosition, 0);

		glGenTextures(1, &gNormalAndShadow);
		glBindTexture(GL_TEXTURE_2D, gNormalAndShadow);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, WIDTH, HEIGHT, 0, GL_RGB, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, gNormalAndShadow, 0);

		
		glGenTextures(1, &gColor);
		glBindTexture(GL_TEXTURE_2D, gColor);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, WIDTH, HEIGHT, 0, GL_RGB, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, gColor, 0);
		

		unsigned int attachments[3] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2 };
		glDrawBuffers(3, attachments);


		//unsigned int rboDepth;
		glGenRenderbuffers(1, &rboDepth);
		glBindRenderbuffer(GL_RENDERBUFFER, rboDepth);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, WIDTH, HEIGHT);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rboDepth);
		// check if framebuffer is complete
		if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
			std::cout << "Framebuffer not complete!" << std::endl;
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}
	void deleteGeoFbo()
	{
		glDeleteTextures(1, &gPosition);
		glDeleteTextures(1, &gNormalAndShadow);
		glDeleteTextures(1, &gColor);
		glDeleteRenderbuffers(1, &rboDepth);
		glDeleteFramebuffers(1, &gBuffer);
	}

	void bind() { glBindFramebuffer(GL_FRAMEBUFFER, gBuffer); };
	void unbind() { glBindFramebuffer(GL_FRAMEBUFFER, 0); };
};

template<int nlayer>
class GLDpFbo 
{
public:
	GLuint rbo, fbo;
	int w, h;
	GLDpFbo() : fbo(0), rbo(0), w(0), h(0)
	{
		for (int i = 0; i < nlayer; i++) { color_layer[i] = 0; }
	}
	~GLDpFbo() { deleteDpFbo(); };

	GLuint color_layer[nlayer];
	GLuint depth[2] = { 0, 0 };
	void createColorLayer()
	{
		glGenTextures(nlayer, color_layer);
		for (int i = 0; i < nlayer; i++)
		{
			glBindTexture(GL_TEXTURE_2D, color_layer[i]);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, w, h, 0, GL_RGB, GL_FLOAT, NULL);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		}
	}

	void createDepthBuffer()
	{
		glGenTextures(2, depth);
		for (int i = 0; i < 2; i++)
		{
			glBindTexture(GL_TEXTURE_2D, depth[i]);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_R16F, w, h, 0, GL_RGB, GL_FLOAT, NULL);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		}
	}

	void createBuffer()
	{
		createColorLayer();
		createDepthBuffer();
	}

	void clearColorLayerBuffer()
	{
		float cleardata[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
		GLuint clfbo = 0;
		glGenFramebuffers(1, &clfbo);
		glBindFramebuffer(GL_FRAMEBUFFER, clfbo);
		for (int i = 0; i < nlayer; i++)
		{
			glBindTexture(GL_TEXTURE_2D, color_layer[i]);
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color_layer[i], 0);
			glClearBufferfv(GL_COLOR, 0, cleardata); 
		}
		glBindTexture(GL_TEXTURE_2D, 0);

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	void clearDepthBuffer()
	{
		float cleardata[1] = { 1.0f};
		GLuint clfbo = 0;
		glGenFramebuffers(1, &clfbo);
		glBindFramebuffer(GL_FRAMEBUFFER, clfbo);
		for (int i = 0; i < 2; i++)
		{
			glBindTexture(GL_TEXTURE_2D, depth[i]);
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, depth[i], 0);
			glClearBufferfv(GL_COLOR, 0, cleardata);
		}
		glBindTexture(GL_TEXTURE_2D, 0);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	void clearDepthBufferAtN(int i)
	{
		float cleardata[1] = { 1.0f };
		GLuint clfbo = 0;
		glGenFramebuffers(1, &clfbo);
		glBindFramebuffer(GL_FRAMEBUFFER, clfbo);

		glBindTexture(GL_TEXTURE_2D, depth[i % 2]);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, depth[i % 2], 0);
		glClearBufferfv(GL_COLOR, 0, cleardata);

		glBindTexture(GL_TEXTURE_2D, 0);
		if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
			std::cout << "Framebuffer not complete!" << std::endl;
		
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	void clearBuffer()
	{
		clearColorLayerBuffer();
		clearDepthBuffer();
	}


	void DpFboBindDepthBuffer(int i)
	{		
		glBindTexture(GL_TEXTURE_2D, depth[i % 2]);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, depth[i % 2], 0);
	}

	void DpFboBindColorBuffer(int i)
	{
		glBindTexture(GL_TEXTURE_2D, color_layer[i]);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, color_layer[i], 0);
	}

	void DpFboBindBuffer(int i)
	{
		glBindFramebuffer(GL_FRAMEBUFFER, fbo);
		DpFboBindDepthBuffer(i);
		DpFboBindColorBuffer(i);
		unsigned int attachments[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
		glDrawBuffers(2, attachments);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}
	

	void createDpFbo(int WIDTH, int HEIGHT)
	{
		glGenFramebuffers(1, &fbo);
		glBindFramebuffer(GL_FRAMEBUFFER, fbo);

		//unsigned int rboDepth;
		glGenRenderbuffers(1, &rbo);
		glBindRenderbuffer(GL_RENDERBUFFER, rbo);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, WIDTH, HEIGHT);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo);

		// check if framebuffer is complete
		if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
			std::cout << "Framebuffer not complete!" << std::endl;
		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		w = WIDTH;
		h = HEIGHT;
		createBuffer();
	}

	void deleteDpFbo()
	{
		glDeleteTextures(nlayer, color_layer);
		glDeleteTextures(2, depth);
		glDeleteRenderbuffers(1, &rbo);
		glDeleteFramebuffers(1, &fbo);
		//
	}

	void bind() { glBindFramebuffer(GL_FRAMEBUFFER, fbo); };
	void unbind() { glBindFramebuffer(GL_FRAMEBUFFER, 0); };
};




Eigen::Matrix3f quaternaion2matrix(const float *q)
{
    double s = Eigen::Map<const Eigen::Vector4f>(q).squaredNorm();
    Eigen::Matrix3f res;

    float *m = res.data();

    m[0] = 1.f - 2.f * (q[1] * q[1] + q[2] * q[2]);
    m[1] = 2.f * (q[0] * q[1] - q[2] * q[3]);
    m[2] = 2.f * (q[2] * q[0] + q[1] * q[3]);

    m[3 + 0] = 2.f * (q[0] * q[1] + q[2] * q[3]);
    m[3 + 1] = 1.f - 2.f * (q[2] * q[2] + q[0] * q[0]);
    m[3 + 2] = 2.f * (q[1] * q[2] - q[0] * q[3]);

    m[6 + 0] = 2.f * (q[2] * q[0] - q[1] * q[3]);
    m[6 + 1] = 2.f * (q[1] * q[2] + q[0] * q[3]);
    m[6 + 2] = 1.f - 2.f * (q[1] * q[1] + q[0] * q[0]);

    return res.transpose();
}

Eigen::Matrix4f perspective(float fovy,  float aspect,  float zNear, float zFar)
{
    assert(aspect > 0);
    assert(zFar > zNear);

    float radf = fovy / 180 * M_PI;
    float tanHalfFovy = tan(radf / 2);
    Eigen::Matrix4f res = Eigen::Matrix4f::Zero();
    res(0, 0) = 1 / (aspect * tanHalfFovy);
    res(1, 1) = 1 / (tanHalfFovy);
    res(2, 2) = -(zFar + zNear) / (zFar - zNear);
    res(3, 2) = -1;
    res(2, 3) = -(2 * zFar * zNear) / (zFar - zNear);
    return res;
}

Eigen::Matrix4f lookAt(const float* eye, const float* center, const float* up)
{
    using Vec = Eigen::RowVector3f;
    using MapVec = Eigen::Map<const Vec>;
    Vec f = (MapVec(center) - MapVec(eye)).normalized();
    Vec u = MapVec(up).normalized();
    Vec s = f.cross(u).normalized();
    u = s.cross(f);

    Eigen::Matrix4f res;
    res.leftCols(3) << s, u, -f, 0, 0, 0;
    res.rightCols(1) << -res.topLeftCorner(3, 3)*MapVec(eye).transpose(), 1;
    return res;
}

Eigen::Matrix4f ortho(const float left, const float right, const float bottom, const float top, const float zNear, const float zFar)
{
	Eigen::Matrix4f res;
	res.setZero();
	res(0, 0) = 2.0f / (right - left);
	res(1, 1) = 2.0f / (top - bottom);
	res(2, 2) = -2.0f / (zFar - zNear);
	res(0, 3) = -(right + left) / (right - left);
	res(1, 3) = -(top + bottom) / (top - bottom);
	res(2, 3) = -(zFar + zNear) / (zFar - zNear);
	res(3, 3) = 1;
	return res;
}

template<typename R = float, int dimension = 3>
struct GLMesh
{
    enum { dim = dimension };
    using MapMat4 = Eigen::Map < Eigen::Matrix < float, 4, 4, Eigen::RowMajor > >;
    using ConstMat4 = Eigen::Map < const Eigen::Matrix < float, 4, 4, Eigen::RowMajor > >;
	typedef std::array<R, 4> vec4;
	typedef std::array<R, dim> vec;

    enum PickingElements { PE_NONE = 0, PE_VERTEX, PE_FACE };
    enum PickingOperations { PO_NONE = 0, PO_ADD, PO_REMOVE };

    struct Mesh
    {
        std::vector<R> X, UV;
		std::vector<R> normal;
        std::vector<int> T;
        size_t nVertex() const { return X.size() / dim; }
        size_t nFace() const { return T.size() / 3; }

    };
    Mesh mesh;
    GLTexture tex;
    static GLTexture colormapTex;
	static GLTexture chessboardTex;
	static GLTexture chessboardTex_background;

	bool isShadowMap = true;
	bool isAo = true;
	//bool IsShowDistortion = false;
	GLFbo depthMapFbo, ssaoFBO, ssaoBlurFBO, resFBO;
	const GLuint SHADOW_WIDTH = 4096, SHADOW_HEIGHT = 4096;		//for high resolution shadow map
	GLGeoFbo geoFbo;
	GLDpFbo<3> dpFbo;
	std::vector<vec> ssaoKernel;	
	GLuint noiseTexture;
	int ssaoKernelSize = 64;



	/*quad in screen*/
	GLuint quadVAO = 0;
	GLuint quadVBO;

	/*border cube*/
	GLuint cubeVAO = 0;
	GLuint cubeVBO = 0;



    int nVertex;
    int nFace;
    GLuint vaoHandle;

    GLArray<R, dim> gpuX;
	GLArray<R, dim> gpuNorm;
    GLArray<int, 3, true> gpuT;
    GLArray<R, 2>   gpuUV;
    GLArray<R, 1>   gpuVertexData;


	//GLArray<R, 1>   gpuVertexDistortion;
    GLArray<R, 1>   gpuFaceData;
    R vtxDataMinMax[2];
	bool vizVtxDistortion;

    std::map<int, vec> constrainVertices;
    std::vector<R> vertexData;
    std::vector<R> faceData;
    std::vector<R> vertexVF;
	

    int actVertex;
    int actFace;
    int actConstrainVertex;

	//vec4 faceColor = { 0.6f, 0.6f, 0.6f, 1.f };
	vec4 faceColor = { 0.8f, 0.8f, 0.8f, 0.3f };
    vec4 edgeColor = { 0.f, 0.f, 0.f, 0.8f };
    vec4 vertexColor = { 1.f, 0.f, 0.f, 1.0f };
    int depthMode = 0;
    float edgeWidth = 0.f;
    float pointSize = 0.f;
	int vp[4] = { 0, 0, 1280, 960 };

    float auxPointSize;
    float vertexDataDrawScale;
    float faceDataDrawScale;
    float VFDrawScale = 0.f;

    float mMeshScale = 1.f;
    float mTextureScale = 0.1f;

    std::vector<int> auxVtxIdxs;

    vec mTranslate = { 0, 0, 0 };
    float mQRotate[4] = { 0,0,0,1 };
    float mViewCenter[3] = { 0, 0, 0 };
    R bbox[dim * 2];

    bool showTexture = false;
    bool drawTargetP2P = true;
	float P2PHandleBlackSize = 1.0;
	float P2PHandleCyanSize = 1.0;
	
	bool isSmooth = true;
	bool expImg = false;
    static GLProgram pickProg, depthTestProg, geometryProj2Screen, pointSet, ssaoProg, screenProg, ssaoBlurProg;

	vec Ambient = { 0.125f, 0.125f, 0.125f };
	vec LightColor = { 0.8f, 0.8f, 0.8f };
	vec LightDirection = { 0.2f, 0.8f, 1.0f };
	float LightAngle =  0;
	GLfloat	Shininess = 30.0f;
	GLfloat	Strength = 0.5f;
	vec HalfVector;


    ~GLMesh() { glDeleteVertexArrays(1, &vaoHandle); }

    GLMesh() :vaoHandle(0), actVertex(-1), actFace(-1),
        actConstrainVertex(-1), vertexDataDrawScale(0.f),
        faceDataDrawScale(0.f), vizVtxDistortion(0), auxPointSize(0.f)
    {
		
	}

	void resize(int w, int h)
	{
		vp[2] = w; vp[3] = h;
		geoFbo.deleteGeoFbo();		
		geoFbo.createGeoFbo(2 * w, 2 * h);
		dpFbo.deleteDpFbo();
		dpFbo.createDpFbo(2 * w, 2 * h);

		ssaoFBO.allocateStorage(2 * w, 2 * h, GL_RED, GL_RGB, GL_NEAREST, GL_REPEAT, GL_COLOR_ATTACHMENT0);
		ssaoBlurFBO.allocateStorage(2 * w, 2 * h, GL_RED, GL_RGB, GL_NEAREST, GL_REPEAT, GL_COLOR_ATTACHMENT0);
		resFBO.allocateStorage(w, h, GL_RGB, GL_RGB, GL_NEAREST, GL_REPEAT, GL_COLOR_ATTACHMENT0, GL_UNSIGNED_BYTE);

	}



    void allocateStorage(int nv, int nf)
    {
        if (nv == nVertex && nf == nFace) return;
        nVertex = nv;
        nFace = nf;

        if (!vaoHandle)  glGenVertexArrays(1, &vaoHandle);

        glBindVertexArray(vaoHandle);

        gpuX.allocateStorage(nv);
        glVertexAttribPointer(0, dim, GLType<R>::val, GL_FALSE, 0, nullptr);
        glEnableVertexAttribArray(0);  // Vertex position

        gpuUV.allocateStorage(nv);
        glVertexAttribPointer(2, 2, GLType<R>::val, GL_FALSE, 0, nullptr);
        glEnableVertexAttribArray(2);  // Texture coords

		gpuNorm.allocateStorage(nv);
		glVertexAttribPointer(3, dim, GLType<R>::val, GL_FALSE, 0, nullptr);
		glEnableVertexAttribArray(3);  // Vertex normals

        gpuVertexData.allocateStorage(nv);
        glVertexAttribPointer(4, 1, GLType<R>::val, GL_FALSE, 0, nullptr);
        glEnableVertexAttribArray(4);  // Vertex data

		gpuT.allocateStorage(nf);
        glBindVertexArray(0);
		
		
		depthMapFbo.allocateStorage(SHADOW_WIDTH, SHADOW_HEIGHT, GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT, GL_NEAREST, GL_REPEAT, GL_DEPTH_ATTACHMENT);
		glDrawBuffer(GL_NONE);
		glReadBuffer(GL_NONE);
		depthMapFbo.unbind();

		

		//Create sample kernel and noise texture
		std::uniform_real_distribution<GLfloat> randomFloats(0.0, 1.0);
		std::default_random_engine generator;
		for (int i = 0; i < ssaoKernelSize; i++)
		{
			vec sample = {randomFloats(generator) * 2.0 - 1.0, randomFloats(generator) * 2.0 - 1.0, randomFloats(generator)};
			Eigen::Map<Eigen::Vector3f>(sample.data()).normalize();
			GLfloat scale = GLfloat(i) / 64.0;
			scale = (0.1f + scale * scale * (1.0f - 0.1f)) * randomFloats(generator);
			Eigen::Map<Eigen::Vector3f>(sample.data()) *= scale;
			ssaoKernel.push_back(sample);
		}

		std::vector<vec> ssaoNoise;
		for (GLuint i = 0; i < 16; i++)
		{
			vec noise = { randomFloats(generator) * 2.0 - 1.0, randomFloats(generator) * 2.0 - 1.0, randomFloats(generator) * 2.0 - 1.0 };
			ssaoNoise.push_back(noise);
		}
		glGenTextures(1, &noiseTexture);
		glBindTexture(GL_TEXTURE_2D, noiseTexture);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, 4, 4, 0, GL_RGB, GL_FLOAT, ssaoNoise[0].data());
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);		
    }

    std::vector<int> getConstrainVertexIds() const {
        std::vector<int> idxs;
        idxs.reserve(constrainVertices.size());
        for (auto it : constrainVertices)   idxs.push_back(it.first);
        return idxs;
    }

    std::vector<R> getConstrainVertexCoords() const {
        std::vector<R> x;
        x.reserve(constrainVertices.size()*dim);
        for (auto it : constrainVertices) {
            x.insert(x.end(), { it.second[0], it.second[1], it.second[2] });
        }
        return x;
    }

    void getTriangulation(int *t) { gpuT.downloadData(t, nFace); }

    void getTriangulation(int ti, int *t) { gpuT.at(ti, t); }

    void getVertex(R *x) { gpuX.downloadData(x, nVertex); }

	void getVertexNorm(R *x) { gpuNorm.downloadData(x, nVertex); }

    void getVertex(int vi, R *x) { gpuX.at(vi, x); }

	void getVertexNorm(int vi, R *x) { gpuNorm.at(vi, x); }

    void setVertex(int vi, const R *x)
    {
        //for (int i = 0; i < dim; i++)    mesh.X[vi*dim + i] = x[i];
        gpuX.setAt(vi, x);
    }

	void setVertexNorm(int vi, const R *x)
	{
		gpuNorm.setAt(vi,x);
	}

    void setConstraintVertices(const int *ids, const R* pos, size_t nc) {
        constrainVertices.clear();
        for (size_t i = 0; i < nc; i++) constrainVertices.insert({ ids[i], vec{ pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2] } });
    }




    void setVertexDataViz(const R* val)
    {
        /*if (val) {
            gpuVertexData.uploadData(val, mesh.nVertex());

            glBindVertexArray(vaoHandle);
            gpuVertexData.bind();
            glVertexAttribPointer(5, 1, GLType<R>::val, GL_FALSE, 0, nullptr);
            glEnableVertexAttribArray(5);

            auto mm = std::minmax_element(val, val + mesh.nVertex());
            vtxDataMinMax[0] = *mm.first;
            vtxDataMinMax[1] = *mm.second;
            prog.bind();
            prog.setUniform("dataMinMax", vtxDataMinMax[0], vtxDataMinMax[1]);
        }
        else {
            glBindVertexArray(vaoHandle);
            glDisableVertexAttribArray(5);
        }*/
    }

    void updateDataVizMinMax()
    {
        /*prog.bind();
        prog.setUniform("dataMinMax", vtxDataMinMax[0], vtxDataMinMax[1]);*/
    }

    template<class MatrixR, class MatrixI>
    void upload(const MatrixR &x, const MatrixI &t, const R *uv)
    {
        upload(x.data(), (int)x.rows(), t.count() ? t.data() : nullptr, (int)t.rows(), uv);
    }

	template<class MatrixR, class MatrixI>
	void upload(const MatrixR &x, const MatrixI &t, const MatrixR &norm, const R *uv)
	{
		upload(x.data(), (int)x.rows(), norm.data(), t.count() ? t.data() : nullptr, (int)t.rows(), uv);
	}
	void upload(const R* x, int nv, const int* t, int nf, const R* uv)
    {
		if (x)
			allocateStorage(x ? nv : nVertex, t ? nf : nFace);
			
        if (x) { gpuX.uploadData(x, nv, false); mesh.X.assign(x, x + nv*dim); }
        if (uv) { gpuUV.uploadData(uv, nv, false); mesh.UV.assign(uv, uv + nv * 2); }
        if (t) { gpuT.uploadData(t, nf, false); mesh.T.assign(t, t + nf * 3); }

        if (x&&t) {
            boundingbox(x, nv, bbox);  // update bounding box for the initialization
            constrainVertices.clear();
            auxVtxIdxs.clear();

            actVertex = -1;
            actFace = -1;
            actConstrainVertex = -1;
        }

    }

	void upload(const R *x, int nv, const R *norm, const int *t, int nf, const R *uv)
	{

		if (x) 
			allocateStorage(x ? nv : nVertex, t ? nf : nFace);

		if (x) { gpuX.uploadData(x, nv, false); mesh.X.assign(x, x + nv*dim); }
		if (uv) { gpuUV.uploadData(uv, nv, false); mesh.UV.assign(uv, uv + nv * 2); }
		if (t) { gpuT.uploadData(t, nf, false); mesh.T.assign(t, t + nf * 3); }
		if (norm) { gpuNorm.uploadData(norm, nv, false); mesh.normal.assign(norm, norm + nv*dim); }

		if (x&&t) {
			boundingbox(x, nv, bbox);  // update bounding box for the initialization
			constrainVertices.clear();
			auxVtxIdxs.clear();

			actVertex = -1;
			actFace = -1;
			actConstrainVertex = -1;

			/*vertexData = std::vector<R>(nv, 0);
			faceData = std::vector<R>(nf, 1);

			gpuVertexData.uploadData(vertexData.data(), nVertex, false);
			gpuFaceData.uploadData(faceData.data(), nFace);*/
		}

	}

    void updateBBox() {
        boundingbox(mesh.X.data(), nVertex, bbox);
    }

	void resetmodelRotation()
	{
		mQRotate[0] = 0.0f;
		mQRotate[1] = 0.0f;
		mQRotate[2] = 0.0f;
		mQRotate[3] = 1.0f;
	}
	void resetViewCenter()
	{
		mViewCenter[0] = 0.0f;
		mViewCenter[1] = 0.0f;
		mViewCenter[2] = 0.0f;
	}

    std::vector<R> baryCenters(const R* X)
    {
        std::vector<R> x(nFace * dim);
        for (int i = 0; i < nFace; i++) {
            const R *px[] = { &X[mesh.T[i * 3] * dim], &X[mesh.T[i * 3 + 1] * dim], &X[mesh.T[i * 3 + 2] * dim] };
            for (int j = 0; j < dim; j++) x[i*dim + j] = (px[0][j] + px[1][j] + px[2][j]) / 3;
        }

        return x;
    }


    float actVertexData() const { return (actVertex >= 0 && actVertex < nVertex) ? vertexData[actVertex] : std::numeric_limits<float>::infinity(); }
    float actFaceData() const { return (actFace >= 0 && actFace < nFace) ? faceData[actFace] : std::numeric_limits<float>::infinity(); }
    void incActVertexData(float pct) { setVertexData(actVertex, (vertexData[actVertex] + 1e-3f) * (1 + pct)); }
    void incActFaceData(float pct) { setFaceData(actFace, (faceData[actFace] + 1e-3f) * (1 + pct)); }

    void setVertexData(int i, R v) {
        MAKESURE(i < nVertex && i >= 0);
        gpuVertexData.setAt(i, &v);
        vertexData[i] = v;
    }

    void setFaceData(int i, R v) {
        MAKESURE(i < nFace && i >= 0);
        gpuFaceData.setAt(i, &v);
        faceData[i] = v;
    }

    void setVertexData(const R *vtxData) { gpuVertexData.uploadData(vtxData, nVertex, false); }
    void setFaceData(const R *vtxData) { gpuFaceData.uploadData(vtxData, nFace, false); }

    bool showWireframe() const { return edgeWidth > 0; }
    bool showVertices() const { return pointSize > 0; }

    R drawscale() const
    {
        R scale0 = 1.9f / std::max(std::max(bbox[dim] - bbox[0], bbox[1 + dim] - bbox[1]), bbox[2 + dim] - bbox[2]);
        return mMeshScale*scale0;
    }

    std::array<R, 16> matMVP(const int *vp, bool offsetDepth = false, bool colmajor = false) const {
        R trans[] = { (bbox[0] + bbox[dim]) / 2, (bbox[1] + bbox[1 + dim]) / 2, (bbox[2] + bbox[2 + dim]) / 2 };
        R ss = drawscale(); 
        R t[] = { mTranslate[0] - trans[0], mTranslate[1] - trans[1], mTranslate[2] - trans[2] };


        using Mat4 = Eigen::Matrix<float, 4, 4, Eigen::RowMajor>;

        Mat4 proj = perspective(45.f, vp[2] / float(vp[3]), 0.1f, 100.0f);
        if(offsetDepth)  proj(2, 2) += 1e-5;
		//float campos[] = { (bbox[0] + bbox[dim]) / 2, (bbox[1] + bbox[1 + dim]) / 2,(bbox[2] + bbox[2 + dim]) / 2+ 4 }, up[] = { 0, 1, 0 };// Head is up (set to 0,-1,0 to look upside-down)
        //float campos[] = { 0, 0, 4 }, up[] = { 0, 1, 0 };// Head is up (set to 0,-1,0 to look upside-down)
		float campos[] = { mViewCenter[0], mViewCenter[1] + 2, mViewCenter[2]+4 }, up[] = { 0, 1, 0 };// Head is up (set to 0,-1,0 to look upside-down)
        Mat4 view = lookAt(campos, mViewCenter, up);

		Mat4 model;
		model << ss*Eigen::Matrix3f::Identity(), Eigen::Map<Eigen::Array3f>(t)*ss,
			0, 0, 0, 1;
		Mat4 Rot;
		Rot << quaternaion2matrix(mQRotate), Eigen::Vector3f::Zero(), 0, 0, 0, 1;
		model = Rot*model;
  
        std::array<R,16> mvp;
		Eigen::Map<Mat4>(mvp.data()) = proj * view * model;
        if (colmajor) Eigen::Map<Mat4>(mvp.data()).transposeInPlace();
        return mvp;
    }

	std::array<R, 16>matModel() const
	{
		R trans[] = { (bbox[0] + bbox[dim]) / 2, (bbox[1] + bbox[1 + dim]) / 2, (bbox[2] + bbox[2 + dim]) / 2 };
		R ss = drawscale();
		R t[] = { mTranslate[0] - trans[0], mTranslate[1] - trans[1], mTranslate[2] - trans[2] };
		using Mat4 = Eigen::Matrix<float, 4, 4, Eigen::RowMajor>;
		
		Mat4 scale;
		scale << ss * Eigen::Matrix3f::Identity(), Eigen::Map<Eigen::Array3f>(t)*ss,
			0, 0, 0, 1;
		Mat4 Rot;
		Rot << quaternaion2matrix(mQRotate), Eigen::Vector3f::Zero(), 0, 0, 0, 1;
		std::array<R, 16> model;
		Eigen::Map<Mat4>(model.data()) = Rot * scale;
		return model;
	}

	std::array<R, 16> matView() const
	{
		using Mat4 = Eigen::Matrix<float, 4, 4, Eigen::RowMajor>;

		float campos[] = { mViewCenter[0], mViewCenter[1] + 2, mViewCenter[2] + 4 }, up[] = { 0, 1, 0 };// Head is up (set to 0,-1,0 to look upside-down)
		std::array<R, 16> view;
		Eigen::Map<Mat4>(view.data()) = lookAt(campos, mViewCenter, up);
		return view;
	}

	std::array<R, 16> matProject(bool offsetDepth = false, float offsetLength = 1e-5) const
	{
		using Mat4 = Eigen::Matrix<float, 4, 4, Eigen::RowMajor>;
		std::array<R, 16> proj;
		Eigen::Map<Mat4>(proj.data()) = perspective(45.f, vp[2] / float(vp[3]), 0.1f, 100.0f);
		if (offsetDepth)  Eigen::Map<Mat4>(proj.data())(2, 2) += offsetLength;
		return proj;
	}

	std::array<R, 9> matNormalTransform(void) const {
		R trans[] = { (bbox[0] + bbox[dim]) / 2, (bbox[1] + bbox[1 + dim]) / 2, (bbox[2] + bbox[2 + dim]) / 2 };
		R ss = drawscale();
		R t[] = { mTranslate[0] - trans[0], mTranslate[1] - trans[1], mTranslate[2] - trans[2] };


		using Mat4 = Eigen::Matrix<float, 4, 4, Eigen::RowMajor>;
		using Mat3 = Eigen::Matrix<float, 3, 3, Eigen::RowMajor>;

		float campos[] = { mViewCenter[0], mViewCenter[1] + 2, mViewCenter[2] + 4 }, up[] = { 0, 1, 0 };// Head is up (set to 0,-1,0 to look upside-down)
		Mat4 view = lookAt(campos, mViewCenter, up);

		Mat4 model;
		model << ss*Eigen::Matrix3f::Identity(), Eigen::Map<Eigen::Array3f>(t)*ss,
			0, 0, 0, 1;
		Mat4 Rot;
		Rot << quaternaion2matrix(mQRotate), Eigen::Vector3f::Zero(), 0, 0, 0, 1;
		model = Rot*model;

		std::array<R, 9> NormalTransform;
		Eigen::Map<Mat3>(NormalTransform.data()) = ((view*model).block<3, 3>(0, 0)).inverse().transpose();
		return NormalTransform;
	}

	std::array<R, 16> matLightSpace(void) const {		
		using Mat4 = Eigen::Matrix<float, 4, 4, Eigen::RowMajor>;
		using Mat3 = Eigen::Matrix<float, 3, 3, Eigen::RowMajor>;
		Mat4 lightProjection = ortho(-8.0f, 8.0f, -8.0f, 8.0f, -8.0f, 8.0f);
		float center0[] = { 0.0f, 0.0f ,0.0f };
		float up[] = { 0.0, 1.0, 0.0 };
		Mat4 lightView = lookAt(LightDirection.data(), center0, up);
		std::array<R, 16> lightSpaceMatrix;
		Eigen::Map<Mat4>(lightSpaceMatrix.data()) = lightProjection * lightView;
		return lightSpaceMatrix;
	}

    void moveInScreen(int x0, int y0, int x1, int y1) {
        float dx = (x1 - x0) * 2.f / vp[2];
        float dy = -(y1 - y0) * 2.f / vp[3];

        mViewCenter[0] -= dx;
        mViewCenter[1] -= dy;
    }

	void renderQuad()
	{
		if (quadVAO == 0)
		{
			float quadVertices[] = {
				// positions        // texture Coords
				-1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
				-1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
				 1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
				 1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
			};

			// setup plane VAO

			glGenVertexArrays(1, &quadVAO);
			glGenBuffers(1, &quadVBO);
			glBindVertexArray(quadVAO);
			glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
			glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
			glEnableVertexAttribArray(1);
			glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
		}
		if (expImg)
			resFBO.bind();
		glBindVertexArray(quadVAO);
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
		glBindVertexArray(0);
		if (expImg)
			resFBO.unbind();
	}

	void RenderCube(GLProgram &Prog)
	{
		GLfloat sq2 = sqrt(2.0f) / 2.0f;
		/*GLfloat backgroundTrans[16] =
		{
			1.0f, 0.0f, 0.0f, 0.0f,
			0.0f, sq2, sq2, 6.0f,
			0.0f, -sq2, sq2, 6.0f,
			0.0f, 0.0f, 0.0f, 1.0f
		};*/
		GLfloat backgroundTrans[16] =
		{
			1.0f, 0.0f, 0.0f, 0.0f,
			0.0f, 1, 0, 9.0f,
			0.0f, 0, 1, 0.0f,
			0.0f, 0.0f, 0.0f, 1.0f
		};
		Prog.setUniform("model", backgroundTrans);
		// Initialize (if necessary)
		if (cubeVAO == 0)
		{
			GLfloat vertices[] = {   
				// Bottom face
				10.0f, -10.0f, -10.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, // top-left
				-10.0f, -10.0f, -10.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, // top-right
				10.0f, -10.0f, 10.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f,// bottom-left
				-10.0f, -10.0f, 10.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, // bottom-right
				10.0f, -10.0f, 10.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, // bottom-left				
				-10.0f, -10.0f, -10.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, // top-right  
			};
			glGenVertexArrays(1, &cubeVAO);
			glGenBuffers(1, &cubeVBO);
			// Fill buffer
			glBindBuffer(GL_ARRAY_BUFFER, cubeVBO);
			glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
			// Link vertex attributes
			glBindVertexArray(cubeVAO);
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (GLvoid*)0);
			glEnableVertexAttribArray(3);
			glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
			glEnableVertexAttribArray(2);
			glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (GLvoid*)(6 * sizeof(GLfloat)));
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			glBindVertexArray(0);
		}
		// Render Cube
		glBindVertexArray(cubeVAO);
		glDrawArrays(GL_TRIANGLES, 0, 6);
		glBindVertexArray(0);
	}

	void render_in_light_space(GLProgram &Prog)
	{
		/*render in light space*/
		Prog.bind();
		//depthTestProg.setUniform("lightSpaceMatrix", matLightSpace().data());
		Prog.setUniform("lightSpaceMatrix", matLightSpace().data());
		glViewport(0, 0, SHADOW_WIDTH, SHADOW_HEIGHT);
		depthMapFbo.bind();
		glClear(GL_DEPTH_BUFFER_BIT);
		RenderCube(Prog);
		Prog.setUniform("model", matModel().data());
		glBindVertexArray(vaoHandle);
		glDrawElements(GL_TRIANGLES, 3 * nFace, GL_UNSIGNED_INT, nullptr); // not GL_INT
		glBindVertexArray(0);
		Prog.unbind();
		depthMapFbo.unbind();

	}

	void geometry_pass(GLProgram &Prog)
	{
		/*geometry pass*/
		
		glViewport(0, 0, 2 * vp[2], 2 * vp[3]);	
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		Prog.bind();
		Prog.setUniform("view", matView().data());
		Prog.setUniform("projection", matProject().data());
		//Prog.setUniform("NormalTrans", matNormalTransform().data());
		Prog.setUniform("lightSpaceMatrix", matLightSpace().data());

		Prog.setUniform("LightDirection", LightDirection.data());
		Prog.setUniform("smooth_mode", int(isSmooth));
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, depthMapFbo.tex);
		Prog.setUniform("shadowMap", 0);
		Prog.setUniform("vizDis", 0);
		//Prog.setUniform("vizTex", 0);
		Prog.setUniform("vizTex", 1);
		glActiveTexture(GL_TEXTURE1);
		chessboardTex_background.bind();
		vec blue = { 128.f / 256.f, 128.f / 256.f, 1.f };
		Prog.setUniform("color", blue.data());
		RenderCube(Prog);
		Prog.setUniform("color", faceColor.data());
		Prog.setUniform("model", matModel().data());
		
		if (vizVtxDistortion)
		{
			Prog.setUniform("vizDis", 1);
			//Prog.setUniform("textScale", 1);
			glActiveTexture(GL_TEXTURE1);
			colormapTex.bind();
			Prog.setUniform("Img", 1);
		}
		else {
			Prog.setUniform("vizDis", 0);
			Prog.setUniform("vizTex", 1);
			Prog.setUniform("textScale", mTextureScale);
			glActiveTexture(GL_TEXTURE1);
			chessboardTex.bind();
			Prog.setUniform("Img", 1);

		}
		glBindVertexArray(vaoHandle);
		glDrawElements(GL_TRIANGLES, 3 * nFace, GL_UNSIGNED_INT, nullptr); // not GL_INT
		glBindVertexArray(0);
		Prog.unbind();

		
		

	}
	void draw_edge(GLProgram &Prog, GLuint& vaohdl)
	{
		
			Prog.bind();
			Prog.setUniform("view", matView().data());
			Prog.setUniform("LightDirection", LightDirection.data());
			Prog.setUniform("model", matModel().data());
			glBindVertexArray(vaohdl);
		
			Prog.setUniform("vizDis", 0);
			Prog.setUniform("vizTex", 0);
			Prog.setUniform("projection", matProject(true).data());
			glLineWidth(edgeWidth);
			Prog.setUniform("color", edgeColor.data());
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			glDrawElements(GL_TRIANGLES, 3 * nFace, GL_UNSIGNED_INT, nullptr); // not GL_INT
			//glEnable(GL_DEPTH_TEST);
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);				
			Prog.unbind();
			glBindVertexArray(0);
	}


	

	void draw_point(GLProgram &Prog)
	{
		Prog.bind();
		Prog.setUniform("view", matView().data());
		Prog.setUniform("model", matModel().data());
		Prog.setUniform("projection", matProject(true, 1e-4).data());
		glBindVertexArray(vaoHandle);

		glEnable(0x8861);		// WTF!!! Must enable this for draw a circle. 

		if (showVertices()) {
			glPointSize(pointSize * mMeshScale);
			Prog.setUniform("color", vertexColor.data());
			glDrawArrays(GL_POINTS, 0, nVertex);
		}
		glBindVertexArray(0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		Prog.bind();
		if (!constrainVertices.empty()) {
			
			const auto idxs = getConstrainVertexIds();
			GLArray<R, dim> consX(getConstrainVertexCoords().data(), idxs.size());

			
			glVertexAttribPointer(0, dim, GLType<R>::val, GL_FALSE, 0, nullptr);
			glEnableVertexAttribArray(0);  // Vertex position

			/**/
			if (P2PHandleCyanSize > 0)
			{
				if (expImg)
					glPointSize((24 * 4) * mMeshScale * P2PHandleCyanSize);
				else
					glPointSize(24 * mMeshScale * P2PHandleCyanSize);
				Prog.setUniform("color", 0.f, 1.f, 1.f, 1.0f);


				if (drawTargetP2P)
					glDrawArrays(GL_POINTS, 0, (GLsizei)idxs.size());


				gpuX.bind();
				glVertexAttribPointer(0, dim, GLType<R>::val, GL_FALSE, 0, nullptr);
				glEnableVertexAttribArray(0);  // Vertex position


				// make sure this is before the next draw
				if (actConstrainVertex >= 0) {
					Prog.setUniform("color", 1.f, 0.f, 1.f, 1.0f);
					const int id = idxs[actConstrainVertex];
					glDrawElements(GL_POINTS, 1, GL_UNSIGNED_INT, &id);
				}
			}
			if (P2PHandleBlackSize > 0)
			{
				if (expImg)
					glPointSize(10 * 4 * mMeshScale * P2PHandleBlackSize);
				else
					glPointSize(10 * mMeshScale * P2PHandleBlackSize);
				Prog.setUniform("projection", matProject(true, 2e-4).data());
				Prog.setUniform("color", 0.f, 0.f, 0.f, 1.0f);
				glDrawElements(GL_POINTS, (GLsizei)idxs.size(), GL_UNSIGNED_INT, idxs.data());
			}


			glPointSize(1.f);
		}
		Prog.unbind();
		glBindVertexArray(0);

	}

	void render_ssao()
	{
		geoFbo.bind();
		//isSmooth ? geometry_pass(geometryProj2Screen) : geometry_pass(geometryProj2Screen_flat);
		geometry_pass(geometryProj2Screen);
		if (showWireframe())
		{
			draw_edge(geometryProj2Screen, vaoHandle);
		}
		draw_point(pointSet);

		geoFbo.unbind();



		/*Create SSAO texture*/
		ssaoFBO.bind();
		//glViewport(0, 0, vp[2], vp[3]);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		ssaoProg.bind();
		ssaoProg.setUniform("samples", ssaoKernel[0].data());
		ssaoProg.setUniform("projection", matProject().data());
		ssaoProg.setUniform("model_scale", drawscale());
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, geoFbo.gPosition);
		ssaoProg.setUniform("gPosition", 0);
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, geoFbo.gNormalAndShadow);
		ssaoProg.setUniform("gNormalAndShadow", 1);
		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_2D, noiseTexture);
		ssaoProg.setUniform("texNoise", 2);
		renderQuad();
		ssaoFBO.unbind();

		/*blur SSAO texture to remove noise*/
		//glViewport(0, 0, vp[2], vp[3]);
		ssaoBlurFBO.bind();
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		ssaoBlurProg.bind();
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, ssaoFBO.tex);
		renderQuad();
		ssaoBlurFBO.unbind();

		/*add light (phong, shadow map and ssao shading model) in screen*/
		glViewport(0, 0, vp[2], vp[3]);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		screenProg.bind();
		screenProg.setUniform("Ambient", Ambient.data());
		screenProg.setUniform("LightColor", LightColor.data());
		screenProg.setUniform("LightDirection", LightDirection.data());
		screenProg.setUniform("halfVector", HalfVector.data());
		screenProg.setUniform("shininess", Shininess);
		screenProg.setUniform("strength", Strength);
		screenProg.setUniform("isShadowMap", int(isShadowMap));
		screenProg.setUniform("isAo", int(isAo));
		//glActiveTexture(GL_TEXTURE0);
		//glBindTexture(GL_TEXTURE_2D, geoFbo.gPosition);
		//screenProg.setUniform("gPosition", 0);
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, geoFbo.gNormalAndShadow);
		screenProg.setUniform("gNormalAndShadow", 1);
		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_2D, geoFbo.gColor);
		screenProg.setUniform("gColor", 2);
		glActiveTexture(GL_TEXTURE3);
		glBindTexture(GL_TEXTURE_2D, ssaoBlurFBO.tex);
		screenProg.setUniform("ssao", 3);
		renderQuad();
	}

	void draw()
	{
		using Vec3 = Eigen::Vector3f;
		Vec3 ld = Eigen::Map<Vec3>(LightDirection.data());
		ld.normalize();
		Vec3 look;
		look << 0, 2, 4;
		look.normalize();
		Eigen::Map<Vec3>(HalfVector.data()) = (ld + look).normalized();
		render_in_light_space(depthTestProg);
		if (!ssaoKernel.empty()) render_ssao();		
    }

	void save_img()
	{
		expImg = true;
		resize(vp[2] * 4, vp[3] * 4);
		glViewport(0, 0, vp[2], vp[3]);
		draw();

		MyImage img;
		img.img_malloc(vp[2], vp[3], 3);
		GLubyte* pPixelData = img.data();   
		glBindTexture(GL_TEXTURE_2D, resFBO.tex);
		glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, pPixelData);

		time_t now_time = time(NULL);  
		tm*  t_tm = localtime(&now_time); 
		img.write(std::string("scrst") + std::to_string(t_tm->tm_min) + std::to_string(t_tm->tm_sec) + std::string(".png"));
		resize(vp[2] / 4, vp[3] / 4);
		expImg = false;
	}

    int moveCurrentVertex(int x, int y)
    {
        if (actVertex < 0 || actVertex >= nVertex) return -1;

        auto mv = matMVP(vp);
        ConstMat4 MVP(mv.data());

        Eigen::Vector4f v; v[3] = 1;
        getVertex(actVertex, v.data());
        v = MVP*v;
        Eigen::Vector4f x1 = MVP.inverse().eval()*Eigen::Vector4f(x / R(vp[2]) * 2 - 1, 1 - y / R(vp[3]) * 2, v[2]/v[3], 1); // Make Sure call eval before topRows
        x1 = x1 / x1[3];

        if (constrainVertices.find(actVertex) != constrainVertices.end()) 
            constrainVertices[actVertex] = { x1[0], x1[1], x1[2] };

        return 0;
    }

	int Search_ConstrainVertices(int x, int y, const int *vp)
	{
		Eigen::Vector4f v; v[3] = 1;

		auto mv = matMVP(vp);
		ConstMat4 MVP(mv.data());

		for (auto iter = constrainVertices.cbegin(); iter != constrainVertices.cend(); iter++)
		{
			getVertex(iter->first, v.data());
			v = MVP*v;
			v = v / v[3];
			//int winX = vp[0] + vp[2] * (v[0] + 1) / 4.0f;
			//int winY = vp[1] + vp[3] * (1-v[1]) / 4.0f;
			int winX = vp[0] + vp[2] * (v[0] + 1) / 2.0f;
			int winY = vp[1] + vp[3] * (1 - v[1]) / 2.0f;
			if (((winX  - x)*(winX  - x) + (winY  - y)*(winY  - y)) < 100.0f)
			{
				return iter->first;
			}
		}
		return -1;
	}

    int pick(int x, int y, int pickElem, int operation)
    {
        int idx = -1;
		glViewport(0, 0, vp[2], vp[3]);
		glEnable(GL_DEPTH_TEST);
		if(pickElem==PE_VERTEX)
			idx = Search_ConstrainVertices(x, y, vp);
        if (idx < 0) {
            ensure(nFace < 16777216, "not implemented for big mesh");

            //glDrawBuffer(GL_BACK);
            glDisable(GL_MULTISAMPLE); // make sure the color will not be interpolated

            glClearColor(1.f, 1.f, 1.f, 0.f);
            glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
            pickProg.bind();
            pickProg.setUniform("MVP", matMVP(vp).data());
            pickProg.setUniform("pickElement", pickElem);  // 0/1 for pick vertex/face
            glBindVertexArray(vaoHandle);


            float pickdist = 15.f;
            glPointSize(pickdist);
            if (pickElem == PE_VERTEX) {
                // write depth for the whole shape
                glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_TRUE);
                glDrawElements(GL_TRIANGLES, 3 * nFace, GL_UNSIGNED_INT, nullptr); // not GL_INT
                glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);				

                pickProg.setUniform("MVP", matMVP(vp,true).data());
                glDrawArrays(GL_POINTS, 0, nVertex);
            }
            else
                glDrawElements(GL_TRIANGLES, 3 * nFace, GL_UNSIGNED_INT, nullptr); // not GL_INT

            unsigned char pixel[4];
            glReadPixels(x, vp[3] - y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, pixel);   // y is inverted in OpenGL
            idx = (pixel[0] + pixel[1] * 256 + pixel[2] * 256 * 256) - 1;   // -1 to get true index, 0 means background

            glBindVertexArray(0);
            pickProg.unbind();
            //glDrawBuffer(GL_FRONT);
            glClearColor(1.f, 1.f, 1.f, 0.f);
			//glEnable(GL_MULTISAMPLE);
        }


        if (pickElem == PE_VERTEX)
            actVertex = idx;
        else if (pickElem == PE_FACE)
            actFace = idx;

        int res = 0; // return how many vertex are added/deleted
        if (idx >= 0) {
            //printf("vertex %d is picked\n", idx);
            if (pickElem == PE_VERTEX) {
                auto it = constrainVertices.find(idx);
                if (it == constrainVertices.end()) {
                    if (operation == PO_ADD && idx < nVertex) {  // add constrain
                        vec v;
                        getVertex(idx, v.data());
                        constrainVertices[idx] = v;
                        res = 1;
                    }
                }
                else if (operation == PO_REMOVE) {
                    constrainVertices.erase(it);
                    res = -1;
                }
            }
        }

        auto i = getConstrainVertexIds();
        auto it = std::find(i.cbegin(), i.cend(), actVertex);
        actConstrainVertex = int((it == i.cend()) ? -1 : (it - i.cbegin()));
        return res;
    }

    static void boundingbox(const R* x, int nv, R *bbox)
    {
        if (nv < 1) {
            printf("empty point set!\n");
            return;
        }

        for (int i = 0; i < dim; i++) bbox[i] = bbox[i + dim] = x[i];

        for (int i = 1; i < nv; i++) {
            for (int j = 0; j < dim; j++) {
                bbox[j] = std::min(bbox[j], x[i*dim + j]);
                bbox[j + dim] = std::max(bbox[j + dim], x[i*dim + j]);
            }
        }
    }

    static void buildShaders() {
		geometryProj2Screen.compileAndLinkAllShadersFromFile("shader/geoInfo.vs", "shader/geoInfo.fs", "shader/geoInfo.gs");	
		pointSet.compileAndLinkAllShadersFromFile("shader/pointSet.vs", "shader/pointSet.fs");
		depthTestProg.compileAndLinkAllShadersFromFile("shader/depthTest.vs", "shader/depthTest.fs");
		ssaoProg.compileAndLinkAllShadersFromFile("shader/screen.vs", "shader/ssao.fs");
		ssaoBlurProg.compileAndLinkAllShadersFromFile("shader/screen.vs", "shader/ssaoBlur.fs");
		screenProg.compileAndLinkAllShadersFromFile("shader/screen.vs", "shader/screen.fs");

		

        //////////////////////////////////////////////////////////////////////////
        pickProg.compileAndLinkAllShadersFromString(
            R"( #version 330
    layout (location = 0) in vec3 VertexPosition;
    flat out int vertexId;
    uniform mat4 MVP;
    void main(){
        gl_Position = MVP*vec4(VertexPosition, 1);
        vertexId = gl_VertexID;
    })",
            R"( #version 330
    uniform int pickElement;
    flat in int vertexId;
    out vec4 FragColor;
    void main(){
        int id = ( (pickElement==0)?vertexId:gl_PrimitiveID ) + 1;
        // Convert the integer id into an RGB color
        FragColor = vec4( (id & 0x000000FF) >>  0, (id & 0x0000FF00) >>  8, (id & 0x00FF0000) >> 16, 255.f)/255.f;
    })");



const unsigned char jetmaprgb[] = {
0,  0,144,
  0,  0,160,
  0,  0,176,
  0,  0,192,
  0,  0,208,
  0,  0,224,
  0,  0,240,
  0,  0,255,
  0, 16,255,
  0, 32,255,
  0, 48,255,
  0, 64,255,
  0, 80,255,
  0, 96,255,
  0,112,255,
  0,128,255,
  0,144,255,
  0,160,255,
  0,176,255,
  0,192,255,
  0,208,255,
  0,224,255,
  0,240,255,
  0,255,255,
 16,255,240,
 32,255,224,
 48,255,208,
 64,255,192,
 80,255,176,
 96,255,160,
112,255,144,
128,255,128,
144,255,112,
160,255, 96,
176,255, 80,
192,255, 64,
208,255, 48,
224,255, 32,
240,255, 16,
255,255,  0,
255,240,  0,
255,224,  0,
255,208,  0,
255,192,  0,
255,176,  0,
255,160,  0,
255,144,  0,
255,128,  0,
255,112,  0,
255, 96,  0,
255, 80,  0,
255, 64,  0,
255, 48,  0,
255, 32,  0,
255, 16,  0,
255,  0,  0,
240,  0,  0,
224,  0,  0,
208,  0,  0,
192,  0,  0,
176,  0,  0,
160,  0,  0,
144,  0,  0,
128,  0,  0 };

        colormapTex.setImage(MyImage((BYTE*)jetmaprgb, sizeof(jetmaprgb) / 3, 1, sizeof(jetmaprgb), 3));  // bug fix, pitch should be w*3
        colormapTex.setClamping(GL_CLAMP_TO_EDGE);
		chessboardTex.setImage(MyImage(std::string("chessboard.jpg")));
		chessboardTex_background.setImage(MyImage(std::string("chessboard_background.png")));
    }
};


typedef GLMesh<> MyMesh;
