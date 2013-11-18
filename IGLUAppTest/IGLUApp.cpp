#include "IGLUApp.h"
#include "SceneHelper.h"

using namespace iglu;
void keys(unsigned char key, int x, int y)
{
	if (key == 'q' || key == 'Q' || key == 27)  // Quit on a 'q' or Escape
		exit(0);
	
}
void mouse( int button, int state, int x, int y )
{
	if ( state == GLUT_DOWN )
		IGLUApp::GetSingleton().GetTrackBall()->SetOnClick( x, y );
	else if (state == GLUT_UP )
		IGLUApp::GetSingleton().GetTrackBall()->Release();
}

void motion( int x, int y )
{
	IGLUApp::GetSingleton().GetTrackBall()->UpdateOnMotion( x, y ); 
	//glutPostRedisplay();
}
//display function
void display()
{
	IGLUApp::GetSingleton().Display();
}

void preprocess()
{
	// Standard OpenGL setup
	glewInit();	
	
}

IGLUApp::IGLUApp()
{
	InitWindow();
	// Create our frame timer
	_frameRate = new IGLUFrameRate( 20 );
	// Create a virtual trackball
	_trackBall = new IGLUTrackball( _winWidth, _winHeight );	
}



void IGLUApp::InitWindow(const char *title,int width, int height)
{
	_winWidth = width;
	_winHeight = height;

	_win = new IGLUWindow( _winWidth, _winHeight, title );
	_win->SetWindowProperties( IGLU_WINDOW_NO_RESIZE |	
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
	_win->Run();
}

IGLUApp::~IGLUApp()
{
}

void IGLUApp::InitScene(const char *sceneFile)
{
	SceneData helper(sceneFile);
	_camPos = vec3(&helper.getCameraData().eye[0]);
	_camAt = vec3(&helper.getCameraData().at[0]);
	_camUp = vec3(&helper.getCameraData().up[0]);
	_fovy = helper.getCameraData().fovy;
	_near = helper.getCameraData().zNear;
	_far = helper.getCameraData().zFar;

	_PM = IGLUMatrix4x4::Perspective(_fovy,_winWidth/_winHeight,_near,_far);	
	_VM = IGLUMatrix4x4::LookAt(_camPos,_camAt,_camUp);
	_MM = IGLUMatrix4x4::Identity();

	//only support for single light now
	_lightPos = vec3(&helper.getLights()[0].pos[0]);
	_lightAt = vec3(&helper.getLights()[0].at[0]);
	_lightColor = vec3(&helper.getLights()[0].color[0]);

	IGLUShaderProgram::Ptr dftShader = new IGLUShaderProgram("shaders/object.vert.glsl",
		"shaders/object.frag.glsl");
	dftShader->SetProgramEnables( IGLU_GLSL_DEPTH_TEST ); 
	dftShader->SetProgramDisables( IGLU_GLSL_BLEND );
	_shaders.push_back(dftShader);

	for (int i=0; i<helper.getObjects().size(); i++)
	{
		SceneObject* obj = helper.getObjects()[i];
		switch (obj->getType())
		{
		case SceneObjType::mesh:
			{
				ObjModelObject* mesh = (ObjModelObject*)obj;
				
				IGLUOBJReader::Ptr objReader  = new IGLUOBJReader( (char*)mesh->getObjFileName().c_str());
				_objReaders.push_back(objReader);
				glm::mat4 trans = mesh->getTransform();						
				_objTransforms.push_back(IGLUMatrix4x4(&trans[0][0]));
				break;
			}
		default:
			break;
		}
	}
	IGLUOBJReader::Ptr objReader  = new IGLUOBJReader( "../../CommonSampleFiles/models/lincoln.obj");
	// We've loaded all our materials, so prepare to use them in rendering
	IGLUOBJMaterialReader::FinalizeMaterialsForRendering();
	
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
		_shaders[0]["project"]         = _PM;
		_shaders[0]["model"]           = _MM * _objTransforms[i];
		_shaders[0]["view"]            = _VM * _trackBall->GetMatrix();
		_shaders[0]["lightIntensity" ] = float(1);
		_shaders[0]["matlInfoTex"]     = IGLUOBJMaterialReader::s_matlCoefBuf;
		_shaders[0]["matlTextures"]    = IGLUOBJMaterialReader::s_matlTexArray;
		_shaders[0]["lightPos"] = _lightPos;
		_shaders[0]["lightColor"] = _lightColor;

		_objReaders[i]->Draw(_shaders[0]);
		_shaders[0]->Disable();
	}



	

	// Draw the framerate on the screen
	char buf[32];
	sprintf( buf, "%.1f fps", _frameRate->EndFrame() );
	IGLUDraw::DrawText( IGLU_FONT_VARIABLE, 0, 0, buf );
}
