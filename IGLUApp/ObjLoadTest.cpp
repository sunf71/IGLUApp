
#include <stdio.h>

// All headers are automatically included from "iglu.h"
#include "iglu.h"

// IGLU classes and constants are all inside the iglu namespace.
using namespace iglu;

// Anything with a ::Ptr is a pointer-like type.  
IGLUOBJReader::Ptr      objReader = 0;
IGLUOBJReader::Ptr objReader2 = 0;
IGLUShaderProgram::Ptr  shader[2] = {0, 0};
IGLUMatrix4x4           proj  = IGLUMatrix4x4::Perspective( 60, 1, 3, 20 );
IGLUMatrix4x4           model = IGLUMatrix4x4::Translate( 0, 0, -11 );
IGLUTrackball::Ptr		ball = 0;

int                     currentShader=0;

// A frame timer we'll use to display a counter on screen
IGLUFrameRate::Ptr      frameTime = 0;


void display ( void )
{
	// Start timing this frame draw
	frameTime->StartFrame();

	// Clear the screen
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

	// Enable() and Disable() are optional.  If not enabled prior to calling
	//    IGLUOBJReader::Draw(), the specified shader will be automatically
	//    enabled (and automatically disabled afterwards).
	// It is good practice to enable & disable manually, to reduce unnecessary
	//    state changes (e.g., when specifying uniform values)
	shader[currentShader]->Enable(); 
	   
		
		if (currentShader == 1)
		{
			shader[1]["project"]         = proj;
			shader[1]["model"]           = IGLUMatrix4x4::Identity();
			shader[1]["view"]            = model * ball->GetMatrix();
			shader[1]["lightIntensity" ] = float(1);
			shader[1]["matlInfoTex"]     = IGLUOBJMaterialReader::s_matlCoefBuf;
			shader[1]["matlTextures"]    = IGLUOBJMaterialReader::s_matlTexArray;
		}
		else
		{
			 shader[currentShader]["project"]   = proj;		
			 shader[currentShader]["modelview"] = model * ball->GetMatrix();
		}
		objReader->Draw( shader[currentShader] );
	shader[currentShader]->Disable(); 

	// Draw the framerate on the screen
	char buf[32];
	sprintf( buf, "%.1f fps", frameTime->EndFrame() );
	IGLUDraw::DrawText( IGLU_FONT_VARIABLE, 0, 0, buf );

	// Swap the GLUT buffers so we're actually show what we drew.
	glutSwapBuffers();
}


void keys(unsigned char key, int x, int y)
{
	if (key == 'q' || key == 'Q' || key == 27)  // Quit on a 'q' or Escape
		exit(0);
	else
		currentShader = (currentShader+1)%2;
}

void idle ( void )
{
	glutPostRedisplay();
}

void mouse( int button, int state, int x, int y )
{
	if ( state == GLUT_DOWN )
		ball->SetOnClick( x, y );
	else if (state == GLUT_UP )
		ball->Release();
}

void motion( int x, int y )
{
	ball->UpdateOnMotion( x, y ); 
	glutPostRedisplay();
}


void reshape( int w, int h )
{
	glViewport( 0, 0, w, h );
	ball->ResizeInteractorWindow( w, h );
	glutPostRedisplay();
}

int main(int argc, char** argv)
{
	// Standard OpenGL/GLUT setup
	glutInit(&argc, argv);
	glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH );
	glutInitWindowSize( 512, 512 );
	glutInitWindowPosition( 20, 20 );
	glutCreateWindow( "Basic IGLU OBJ Loader" );
	glewInit();
	glutDisplayFunc( display );
	glutReshapeFunc( reshape );
	glutIdleFunc( idle );
	glutKeyboardFunc( keys );
	glutMouseFunc( mouse );
	glutMotionFunc( motion );

	// Create our frame timer
	frameTime = new IGLUFrameRate( 20 );

	// Load the texture with IGLU
	printf("(+) Loading object...\n" );
	objReader  = new IGLUOBJReader( "../../CommonSampleFiles/models/lincoln.obj");


	// We've loaded all our materials, so prepare to use them in rendering
	IGLUOBJMaterialReader::FinalizeMaterialsForRendering();
	// Create a virtual trackball
	ball = new IGLUTrackball( 512, 512 );

	// Load a shader for the object
	printf("(+) Loading shaders...\n" );
	shader[0] = new IGLUShaderProgram( "test.vert.glsl", "test.frag.glsl" );
	//shader[1] = new IGLUShaderProgram( "test.vert.glsl", "test2.frag.glsl" );
	shader[1] = new IGLUShaderProgram("shaders/test.vert.glsl", "shaders/test.frag.glsl"  );
	// Start the GLUT interaction loop
	glutMainLoop();
	return 0;
}
