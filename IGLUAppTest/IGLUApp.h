#ifndef IGLUAPP_H
#define IGLUAPP_H

#include <stdio.h>
// All headers are automatically included from "iglu.h"
#include "iglu.h"
// IGLU classes and constants are all inside the iglu namespace.
#include <string>
#include <vector>
using namespace iglu;

//callbacks
void display();
void mouse( int button, int state, int x, int y );
void motion( int x, int y );
void keys(unsigned char key, int x, int y);
void preprocess();

class IGLUApp
{
protected:
	IGLUApp();
	virtual ~IGLUApp();
	 IGLUApp(IGLUApp const&);              // Don't Implement
     void operator=(IGLUApp const&); // Don't implement
public :
	static IGLUApp& GetSingleton()
	{
		static IGLUApp  _app;
		return _app;
	}
	
	IGLUTrackball::Ptr GetTrackBall()
	{
		return _trackBall;
	}
	IGLUFrameRate::Ptr GetFrameRate()
	{
		return _frameRate;
	}
	void Run();
	virtual void Display();
	virtual void InitScene(const char *sceneFile);
	IGLUMatrix4x4& GetViewMatrix()
	{
		return _VM;
	}
	IGLUMatrix4x4& GetProjectionMatrix()
	{
		return _PM;
	}
	IGLUMatrix4x4& GetModelMatrix()
	{
		return _MM;
	}

	

protected:
	
	virtual void InitWindow(const char *title = "IGLUApp", int width=512, int height=512);	
	
private:
	int _winWidth,_winHeight;
	std::string _sceneFileName;
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
	IGLUTrackball::Ptr _trackBall;
	IGLUFrameRate::Ptr  _frameRate;

	//scene Data
	vec3 _camPos, _camAt, _camUp;
	float _fovy,_aspectRatio,_near,_far;
	vec3 _lightPos, _lightAt, _lightColor;
	//mvp matrix
	IGLUMatrix4x4 _PM,_VM,_MM;
	
	
};

#endif