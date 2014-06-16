#ifndef IGLUAPP_H
#define IGLUAPP_H
#include "Frustum.h"
#include <stdio.h>
// All headers are automatically included from "iglu.h"
#include "iglu.h"
// IGLU classes and constants are all inside the iglu namespace.
#include <string>
#include <vector>
#include "Camera.h"
#include "SceneHelper.h"

using namespace iglu;

//callbacks
void display();
void mouse( int button, int state, int x, int y );
void motion( int x, int y );
void keys(unsigned char key, int x, int y);
void preprocess();

class IGLUApp
{
public:
	IGLUApp(const char* sceneFile);
	IGLUApp():_winWidth(512),_winHeight(512),_sceneData(NULL),_initialized(false)
	{}
	virtual ~IGLUApp();
	 IGLUApp(IGLUApp const&);              // Don't Implement
     void operator=(IGLUApp const&); // Don't implement
public :
	Camera* GetCamera()
	{
		return _camera;
	}
	IGLUFrameRate::Ptr GetFrameRate()
	{
		return _frameRate;
	}
	void Run();
	virtual void Display();
	
	virtual void InitScene();
	
protected:
	virtual void InitApp()
	{
		InitShaders();
		InitScene();
		
	}
	virtual void InitWindow(const char *title = "IGLUApp", int width=512, int height=512);	
	virtual void InitShaders();

	int _winWidth,_winHeight;
	
	//main window
	IGLUWindow::Ptr        _win;
	//ui window
    IGLUWidgetWindow::Ptr   _uiWin;
	// vertex arrays
	std::vector<IGLUVertexArray::Ptr> _vaPtrs;
	// object readers
	std::vector<IGLUOBJReader::Ptr> _objReaders;
	std::vector<IGLUMatrix4x4> _objTransforms;
	// shaders
	std::vector<IGLUShaderProgram::Ptr > _shaders;
	
	IGLUFrameRate::Ptr  _frameRate;

	
	vec3 _lightPos, _lightAt, _lightColor;
	
	
	SceneData* _sceneData;
	bool _initialized;

	Camera * _camera;
};

#endif