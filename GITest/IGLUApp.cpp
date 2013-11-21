#include "IGLUApp.h"
#include "Frustum.h"
#include "glmModel.h"

extern IGLUApp* app;
using namespace iglu;
void keys(unsigned char key, int x, int y)
{
	if (key == 'q' || key == 'Q' || key == 27)  // Quit on a 'q' or Escape
		exit(0);
	else if (key == 'w')
		app->GetCamera()->MoveForward(0.1);
	else if (key == 's')
		app->GetCamera()->MoveBackward(0.1);
	else if (key == 'a')
		app->GetCamera()->MoveLeft(0.1);
	else if (key == 'd')
		app->GetCamera()->MoveRight(0.1);
	else if (key == 'r')
		app->GetCamera()->ResetView();
}
void mouse( int button, int state, int x, int y )
{
	if ( state == GLUT_DOWN )
		app->GetCamera()->SetOnClick( x, y );
	else if (state == GLUT_UP )
		app->GetCamera()->Release();
}

void motion( int x, int y )
{
	app->GetCamera()->UpdateOnMotion(x,y); 
	//glutPostRedisplay();
}
//display function
void display()
{
	app->Display();
}

void preprocess()
{
	// Standard OpenGL setup
	glewInit();	
	
}

IGLUApp::IGLUApp(const char* sceneFile):_initialized(false)
{
	_sceneData = new SceneData(sceneFile);
	SceneData& helper = *_sceneData;
	_camera = new Camera(helper.getCameraData());
	
	 _winWidth = helper.getCameraData().resX >0 ? helper.getCameraData().resX : 512;
	 _winHeight = helper.getCameraData().resY >0 ? helper.getCameraData().resY :512;

	

	//only support for single light now
	_lightPos = vec3(&helper.getLights()[0].pos[0]);
	_lightAt = vec3(&helper.getLights()[0].at[0]);
	_lightColor = vec3(&helper.getLights()[0].color[0]);

	InitWindow(sceneFile,_winWidth,_winHeight);
	
	
	// Create our frame timer
	_frameRate = new IGLUFrameRate( 20 );
	
}



void IGLUApp::InitWindow(const char *title,int width, int height)
{
	

	_win = new IGLUWindow( _winWidth, _winHeight, title );
	_win->SetWindowProperties( 	
								IGLU_WINDOW_DOUBLE |
								IGLU_WINDOW_REDRAW_ON_IDLE								
								); 
	_win->SetDisplayCallback( display );  
	_win->SetKeyboardCallback(keys);
	_win->SetMouseButtonCallback(mouse);
	_win->SetActiveMotionCallback(motion);	
	_win->SetPreprocessOnGLInit(preprocess);
	_win->SetIdleCallback( IGLUWindow::NullIdle );	
	_win->CreateWindow( );     
	
	// Create a widget window to allow us to interact/change our IGLUVariables
	//_uiWin = new IGLUWidgetWindow( 300, 220, "UI Widget Window" );
	/*_uiWin->AddWidget( &innerTessFactor );
	_uiWin->AddWidget( &outerTessFactor );
	_uiWin->AddWidget( &wireframe );	
	_uiWin->AddWidget( &light );	*/
	// Tell our main window about the widget window... 
	//_win->SetWidgetWindow( _uiWin );
}



void IGLUApp::Run()
{
	if (!_initialized)
	{
		InitApp();
		_initialized = true;
	}
	_win->Run();
}

IGLUApp::~IGLUApp()
{
	_shaders.clear();
	_objReaders.clear();
	_objTransforms.clear();
	delete _win;	
	delete _frameRate;
	delete _sceneData;
	delete _camera;
}
void IGLUApp::InitShaders()
{
	IGLUShaderProgram::Ptr dftShader = new IGLUShaderProgram("shaders/object.vert.glsl",
		"shaders/object.frag.glsl");
	dftShader->SetProgramEnables( IGLU_GLSL_DEPTH_TEST | IGLU_GLSL_BLEND); 
	//dftShader->SetProgramDisables( IGLU_GLSL_BLEND );
	_shaders.push_back(dftShader);
}


void IGLUApp::InitScene()
{
	for (int i=0; i<_sceneData->getObjects().size(); i++)
	{
		SceneObject* obj = _sceneData->getObjects()[i];
		switch (obj->getType())
		{
		case SceneObjType::mesh:
			{
				ObjModelObject* mesh = (ObjModelObject*)obj;
				GLMmodel* model = glmReadOBJ(mesh->getObjFileName().c_str());
				IGLUOBJReader::Ptr objReader  = new IGLUOBJReader( model,IGLU_OBJ_COMPACT_STORAGE);
				//IGLUOBJReader::Ptr objReader  = new IGLUOBJReader( (char*)mesh->getObjFileName().c_str(),IGLU_OBJ_COMPACT_STORAGE);
				_objReaders.push_back(objReader);
				glm::mat4 trans = mesh->getTransform();
			/*	std::cout<< trans;*/
				_objTransforms.push_back(IGLUMatrix4x4(&trans[0][0]));				
				delete model;
				break;
			}
		default:
			break;
		}
	}
	
	// We've loaded all our materials, so prepare to use them in rendering
	IGLUOBJMaterialReader::FinalizeMaterialsForRendering(IGLU_TEXTURE_REPEAT);
	
	
}

void IGLUApp::Display()
{
	
	// Start timing this frame draw
	_frameRate->StartFrame();
	glDisable(GL_BLEND);
	// Clear the screen
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
	
	for(int i=0; i<_objReaders.size(); i++)
	{
		_shaders[0]->Enable();
		_shaders[0]["project"]         = _camera->GetProjectionMatrix();
		_shaders[0]["model"]           = _objTransforms[i];
		_shaders[0]["view"]            = _camera->GetViewMatrix();
		_shaders[0]["lightIntensity" ] = float(1);
		_shaders[0]["matlInfoTex"]     = IGLUOBJMaterialReader::s_matlCoefBuf;
		_shaders[0]["matlTextures"]    = IGLUOBJMaterialReader::s_matlTexArray;
		//_shaders[0]["lightPos"] = _lightPos;
		_shaders[0]["lightColor"] = _lightColor;	
		_objReaders[i]->Draw(_shaders[0]);
		_shaders[0]->Disable();

	}

	// Draw the framerate on the screen
	char buf[32];
	sprintf( buf, "%.1f fps", _frameRate->EndFrame() );
	IGLUDraw::DrawText( IGLU_FONT_VARIABLE, 0, 0, buf );
}
