#define _CRT_SECURE_NO_WARNINGS
#define _SCL_SECURE_NO_WARNINGS

#define FREEGLUT_STATIC
#include "gl_core_3_3.h"
#include <GL/glut.h>
#include <GL/freeglut_ext.h>

#define TW_STATIC
#include <AntTweakBar.h>


#include <ctime>
#include <memory>
#include <vector>
#include <string>
#include <cstdlib>
#include <thread>

#include "glprogram.h"
#include "MyImage.h"
#include "VAOImage.h"
#include "VAOMesh.h"
#include "trackball.h"

#include "matlab_utils.h"
#include "matlabDeformer.h"

GLProgram MyMesh::pickProg, MyMesh::depthTestProg, MyMesh::geometryProj2Screen, MyMesh::pointSet, MyMesh::ssaoProg, MyMesh::ssaoBlurProg, MyMesh::screenProg;
GLTexture MyMesh::colormapTex, MyMesh::chessboardTex, MyMesh:: chessboardTex_background;


std::shared_ptr<Deformer> deformer;

MyMesh M;
int actPrimType = MyMesh::PE_VERTEX;

bool showATB = true;

using MatX3f = Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>;
using MatX3i = Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>;
using MatX2f = Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor>;

//int nIteration = 1;
float BackgroundColor[3] = { 1.0f, 1.0f, 1.0f };
int model_id = 0;
std::vector<std::string> model_names;    //const char *[] = { 'hand', 'horse' };

std::vector<std::string> method_names;    //const char *[] = { "newton", ... };


int method = -1;

double w_smooth = 50;




void loadMeshFromMatlab()
{
    /*load vertices, faces, texcoords and vertices-norm*/
    MatX3f X;
    MatX3i T;	
	MatX2f texCoord;
    matlab2eigen("single(viewX)", X, true);
    matlab2eigen("int32(viewT-1)", T, true);	
	matlab2eigen("single(para_Coord)", texCoord, true);
    MatX3f N;
    matlab2eigen("single(normX)", N, true);

	M.upload(X.data(), X.rows(), N.data(), T.data(), T.rows(), texCoord.data());
}


void meshDeform()
{
	if (deformer)
	{
		deformer->deform();
	}
}



int mousePressButton;
int mouseButtonDown;
int mousePos[2];


void display()
{
	glClearColor(BackgroundColor[0], BackgroundColor[1], BackgroundColor[2], 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);




    glViewport(0, 0, M.vp[2], M.vp[3]);
    M.draw();

    if (showATB) TwDraw();
    glutSwapBuffers();

    glFlush();
}

void onKeyboard(unsigned char code, int x, int y)
{
    if (!TwEventKeyboardGLUT(code, x, y)) {
        switch (code) {
        case 17:
            exit(0);
        case 'f':
            glutFullScreenToggle();
            break;
        case ' ':
            showATB = !showATB;
            break;
        }
    }

    glutPostRedisplay();
}

void onMouseButton(int button, int updown, int x, int y)
{
    if (!showATB || !TwEventMouseButtonGLUT(button, updown, x, y)) {
        mousePressButton = button;
        mouseButtonDown = updown;

        if (updown == GLUT_DOWN) {
            if (button == GLUT_LEFT_BUTTON) {
                if (glutGetModifiers()&GLUT_ACTIVE_CTRL) {
                }
                else {
                    int r = M.pick(x, y, M.PE_VERTEX, M.PO_ADD);
                }
            }
            else if (button == GLUT_RIGHT_BUTTON) {
				matlabEval("Deformation_Converged = 0;");
                M.pick(x, y, M.PE_VERTEX, M.PO_REMOVE);
				meshDeform();
            }
        }
        else { // updown == GLUT_UP
            if (button == GLUT_LEFT_BUTTON);
        }

        mousePos[0] = x;
        mousePos[1] = y;
    }

    glutPostRedisplay();
}


void onMouseMove(int x, int y)
{
    if (!showATB || !TwEventMouseMotionGLUT(x, y)) {
        if (mouseButtonDown == GLUT_DOWN) {
            if (mousePressButton == GLUT_MIDDLE_BUTTON) {
                M.moveInScreen(mousePos[0], mousePos[1], x, y);
            }
            else if (mousePressButton == GLUT_LEFT_BUTTON) {
                if (!M.moveCurrentVertex(x, y)) {
					matlabEval("Deformation_Converged = 0;");
                    meshDeform();
					display();
                }
                else {
                    const float s[2] = { 2.f / M.vp[2], 2.f / M.vp[3] };
                    auto r = Quat<float>(M.mQRotate)*Quat<float>::trackball(x*s[0] - 1, 1 - y*s[1], s[0]*mousePos[0] - 1, 1 - s[1]*mousePos[1]);
                    std::copy_n(r.q, 4, M.mQRotate);
                }
            }
        }
    }

    mousePos[0] = x; mousePos[1] = y;

    glutPostRedisplay();
}


void onMouseWheel(int wheel_number, int direction, int x, int y)
{
    M.mMeshScale *= direction > 0 ? 1.1f : 0.9f;
    glutPostRedisplay();
}

int initGL(int argc, char **argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_MULTISAMPLE);
    glutInitWindowSize(1280, 960);
    glutInitWindowPosition(200, 50);
    glutCreateWindow(argv[0]);

    // !Load the OpenGL functions. after the opengl context has been created
    if (ogl_LoadFunctions() == ogl_LOAD_FAILED)
        return -1;

	glClearColor(BackgroundColor[0], BackgroundColor[1], BackgroundColor[2], 0);

	glutReshapeFunc([](int w, int h) { M.resize(w, h); /*viewport[2] = w; viewport[3] = h;*/ TwWindowSize(w, h); });
    glutDisplayFunc(display);
    glutKeyboardFunc(onKeyboard);
    glutMouseFunc(onMouseButton);
    glutMotionFunc(onMouseMove);
    glutMouseWheelFunc(onMouseWheel);
    glutCloseFunc([]() {exit(0); });
    return 0;
}

void loadMesh(std::string modelname)
{
    string2matlab("model_name", modelname);
    matlabEval("loadMesh");
    loadMeshFromMatlab();
    deformer.reset();
    deformer.reset(new MatlabDeformer(M));
}




void createTweakbar()
{

    TwBar *bar = TwGetBarByName("MeshViewer");
    if (bar)    TwDeleteBar(bar);

    //Create a tweak bar
    bar = TwNewBar("MeshViewer");
    TwDefine(" MeshViewer size='220 250' color='0 128 255' text=dark alpha=128 position='5 5'"); // change default tweak bar size and color
    
	model_names = matlab2strings("models");
	loadMesh(model_names[0]);

	TwType modelType = TwDefineEnumFromString("Models", catStr(model_names).c_str());
	/*TwAddVarRW(bar, "Models", modelType, &model, " ");*/
	TwAddVarCB(bar, "Shape", modelType,
		[](const void *v, void *d) {
		model_id = *(int*)v;
		if (model_id < model_names.size()) { loadMesh(model_names[model_id]);}
	},
		[](void *v, void *) { *(int*)v = model_id; },
		nullptr, " ");

	//deformer.reset(new MatlabDeformer(M));

    TwAddVarRO(bar, "#Vertex", TW_TYPE_INT32, &M.nVertex, " group='Mesh View'");
    TwAddVarRO(bar, "#Face", TW_TYPE_INT32, &M.nFace, " group='Mesh View'");
    TwAddVarRW(bar, "Point Size", TW_TYPE_FLOAT, &M.pointSize, " group='Mesh View' ");
    TwAddVarRW(bar, "Edge Width", TW_TYPE_FLOAT, &M.edgeWidth, " group='Mesh View' ");
	TwAddVarRW(bar, "Texture Scale", TW_TYPE_FLOAT, &M.mTextureScale, "min=0.01 max=10 step=0.5 group ='Mesh View' help='Texture Scale'");
	TwAddVarRW(bar, "Smooth Mode", TW_TYPE_BOOLCPP, &M.isSmooth, " group='Mesh View'");

    for (int i = 0; i < method_names.size(); i++)
        matlabEval(method_names[i] + std::string(".set_p2p_weight(p2p_weight);"), false);

    /*TwType methodType = TwDefineEnumFromString("Method", catStr(method_names).c_str());
    TwAddVarCB(bar, "Method", methodType,
        [](const void* v, void* d) {
            method = *(int*)v;
            matlabEval(method_names[method] + std::string(".ResetPointConstraints"), false);
            meshDeform();
        },
        [](void* v, void*) { *(int*)v = method; },
            nullptr, "group='Deformer'");*/

    TwAddVarCB(bar, "enable GPU deformer", TW_TYPE_BOOLCPP,
        [](const void* v, void* d) {
                matlabEval(method_names[method] + std::string(".setGPUdeformer;"), false);
                matlabEval(method_names[method] + std::string(".pre_numPoint_constraints = 0;"), false);
        },
        [](void* v, void* d) {
                *(bool*)(v) = matlab2bool(method_names[method] + ".hasgpu", false, true);
        },
            nullptr, "group='Deformer'");
    //////////////////////////////////////////////////////////////////////////


    TwAddVarCB(bar, "P2P weight", TW_TYPE_FLOAT,
        [](const void* v, void*) { scalar2matlab("p2p_weight", *(const float*)(v));
    matlabEval(method_names[method] + std::string(".ResetPointConstraints;"), false);
    for (int i = 0; i < method_names.size(); i++)
        matlabEval(method_names[i] + std::string(".set_p2p_weight(p2p_weight);"), false);
        },
        [](void* v, void*) { *(float*)(v) = matlab2scalar("p2p_weight"); },
            nullptr, " min=0 ");

    TwAddButton(bar, "Reset View", [](void* d) {
        M.updateBBox();
        M.mMeshScale = 1.f;
        M.mTranslate.assign(0.f);
        //deformerptr(d)->M.resetmodelRotation();
        M.resetViewCenter();
        }, nullptr, "group='Deformer'");

    TwAddButton(bar, "Reset Shape", [](void* d) {
        M.constrainVertices.clear();
        M.actConstrainVertex = -1;
        deformer->resetDeform();
        }, nullptr, " group='Deformer' key=r ");

    TwAddVarCB(bar, "Pause", TW_TYPE_BOOLCPP,
        [](const void* v, void* d) {  deformer->needIteration = !*(bool*)(v); },
        [](void* v, void* d) { *(bool*)(v) = !deformer->needIteration; },
        nullptr, " group='Deformer' key=i ");
}

int main(int argc, char *argv[])
{
	
    if (initGL(argc, argv)) {
        fprintf(stderr, "!Failed to initialize OpenGL!Exit...");
        exit(-1);
    }
	getMatEngine().connect("");
    MyMesh::buildShaders();

    //////////////////////////////////////////////////////////////////////////
    TwInit(TW_OPENGL_CORE, NULL);
    //Send 'glutGetModifers' function pointer to AntTweakBar;
    //required because the GLUT key event functions do not report key modifiers states.
    TwGLUTModifiersFunc(glutGetModifiers);
    glutSpecialFunc([](int key, int x, int y) { TwEventSpecialGLUT(key, x, y); glutPostRedisplay(); }); // important for special keys like UP/DOWN/LEFT/RIGHT ...
    TwCopyStdStringToClientFunc([](std::string& dst, const std::string& src) {dst = src; });

    

    //////////////////////////////////////////////////////////////////////////
    //atexit([] { fprintf(stdout, "exiting...");  TwDeleteAllBars();  TwTerminate(); fprintf(stdout, "exited."); });
	atexit([] { deformer.reset();  TwDeleteAllBars();  TwTerminate(); fprintf(stdout, "exited."); });

    glutTimerFunc(1, [](int) {  
		matlabEval("list_models", false);
        matlabEval("addfolders;");
		createTweakbar();
        
    }, 
        0);

    glutTimerFunc(5, [](int) {
        display();
        },
        0);

    glutIdleFunc([]() { 
        using namespace std::literals::chrono_literals;
        if (deformer && deformer->needIteration && !deformer->converged() && M.getConstrainVertexIds().size()) 
        { meshDeform();  glutPostRedisplay(); }
        else std::this_thread::sleep_for(50ms); }
    );


    //////////////////////////////////////////////////////////////////////////
    glutMainLoop();

    return 0;
}
