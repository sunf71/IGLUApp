#include <stdio.h>

// All headers are automatically included from "iglu.h"
#include "iglu.h"
#include "teapotdata.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform2.hpp>

// IGLU classes and constants are all inside the iglu namespace.
using namespace iglu;

// These are handles to the two windows, should you need to modify their
//   behavior after window creation.
IGLUWindow::Ptr         myWin = 0;
IGLUWidgetWindow::Ptr   uiWin = 0;

// Anything with a ::Ptr is a pointer-like type.  
IGLUVertexArray::Ptr     vertArray = 0;
IGLUVertexArray::Ptr teapotVA = 0;
IGLUShaderProgram::Ptr  shader[3] = {0, 0, 0};
IGLUMatrix4x4 view = IGLUMatrix4x4::LookAt(vec3(4.25f * cos(45.f),3.0f,4.25f * sin(45.f)),vec3(0,0,0),vec3(0,1,0));
IGLUMatrix4x4           proj  = IGLUMatrix4x4::Perspective( 60, 1, 3, 20 );
IGLUMatrix4x4           model = IGLUMatrix4x4::Translate( 0, 0, -11 );
glm::mat4	viewport;
IGLUTrackball::Ptr		ball = 0;

int                     currentShader=0;

IGLUFloat innerTessFactor ( 16.f, IGLURange<float>(1.0,64.0), 1.f,  "innerTessFactor:" );
IGLUFloat  outerTessFactor ( 16.f, IGLURange<float>(1.0,64.0), 1.f,  "outerTessFactor:" );
IGLUBool wireframe(true,"wireframe");
IGLUBool light(true,"light");
// A frame timer we'll use to display a counter on screen
IGLUFrameRate::Ptr      frameTime = 0;


void display ( void )
{
	// Start timing this frame draw
	frameTime->StartFrame();

	// Clear the screen
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
	glEnable(GL_DEPTH_TEST);
	// enable / disable wireframe
	glPolygonMode( GL_FRONT_AND_BACK, wireframe ? GL_LINE : GL_FILL);


	// Enable() and Disable() are optional.  If not enabled prior to calling
	//    IGLUOBJReader::Draw(), the specified shader will be automatically
	//    enabled (and automatically disabled afterwards).
	// It is good practice to enable & disable manually, to reduce unnecessary
	//    state changes (e.g., when specifying uniform values)
	shader[currentShader]->Enable(); 	
	if (currentShader < 2)
	{
		shader[currentShader]["MVP"]         = proj * model * ball->GetMatrix();
		shader[currentShader]["ModelView"]           = model * ball->GetMatrix();		
		shader[currentShader]["light" ] = light ? 1 : 0;
		shader[currentShader]["innerTessFactor"] = innerTessFactor;
		shader[currentShader]["outerTessFactor"] = outerTessFactor;
		glPatchParameteri( GL_PATCH_VERTICES, 1);
		vertArray->DrawArrays(GL_PATCHES,0,1);
	}
	else
	{
		glPolygonMode( GL_FRONT_AND_BACK, GL_FILL);
        
		shader[currentShader]["TessLevel"] = (int)innerTessFactor;
		IGLUMatrix4x4 mv = model * ball->GetMatrix();	
		shader[currentShader]["ModelViewMatrix"] = mv;
		shader[currentShader]["NormalMatrix"] = glm::mat3(glm::vec3(mv[0])
		,glm::vec3(mv[1]),glm::vec3(mv[2]));
		shader[currentShader]["MVP"] = proj * mv;
		float w2 = 512 / 2.0f;
		float h2 = 512 / 2.0f;
		viewport = glm::mat4( glm::vec4(w2,0.0f,0.0f,0.0f),
			glm::vec4(0.0f,h2,0.0f,0.0f),
			glm::vec4(0.0f,0.0f,1.0f,0.0f),
			glm::vec4(w2+0, h2+0, 0.0f, 1.0f));
		//IGLUMatrix4x4 vpm(&viewport[0][0]);
		shader[currentShader]["ViewportMatrix"] = viewport;

		glPatchParameteri(GL_PATCH_VERTICES, 16);
		teapotVA->DrawArrays(GL_PATCHES,0,512);
	}
	shader[currentShader]->Disable(); 
	glPolygonMode( GL_FRONT_AND_BACK, GL_FILL);
	// Draw the framerate on the screen
	/*char buf[32];
	sprintf( buf, "%.1f fps", frameTime->EndFrame() );
	IGLUDraw::DrawText( IGLU_FONT_VARIABLE, 0, 0, buf );*/

	// Swap the GLUT buffers so we're actually show what we drew.
	
}


void keys(unsigned char key, int x, int y)
{
	if (key == 'q' || key == 'Q' || key == 27)  // Quit on a 'q' or Escape
		exit(0);
	else if (key == 't')
		currentShader = 2;
	else if (key == 'w')
		model *= IGLUMatrix4x4::Translate(0,0,0.1);
	else if (key == 's')
		model *= IGLUMatrix4x4::Translate(0,0,-0.1);
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
	//glutPostRedisplay();
}


void reshape( int w, int h )
{
	glViewport( 0, 0, w, h );
	ball->ResizeInteractorWindow( w, h );

	
	//glutPostRedisplay();
}
//VBOTeapotPatch::VBOTeapotPatch()
//{
//    int verts = 32 * 16;
//    float * v = new float[ verts * 3 ];
//
//    glGenVertexArrays( 1, &vaoHandle );
//    glBindVertexArray(vaoHandle);
//
//    unsigned int handle;
//    glGenBuffers(1, &handle);
//
//    generatePatches( v );
//
//    glBindBuffer(GL_ARRAY_BUFFER, handle);
//    glBufferData(GL_ARRAY_BUFFER, (3 * verts) * sizeof(float), v, GL_STATIC_DRAW);
//    glVertexAttribPointer( 0, 3, GL_FLOAT, GL_FALSE, 0, 0 );
//    glEnableVertexAttribArray(0);  // Vertex position
//
//    delete [] v;
//
//    glBindVertexArray(0);
//}


void getPatch( int patchNum, glm::vec3 patch[][4], bool reverseV )
{
	
    for( int u = 0; u < 4; u++) {          // Loop in u direction
        for( int v = 0; v < 4; v++ ) {     // Loop in v direction
            if( reverseV ) {
                patch[u][v] = glm::vec3(
                        Teapot::cpdata[Teapot::patchdata[patchNum][u*4+(3-v)]][0],
                        Teapot::cpdata[Teapot::patchdata[patchNum][u*4+(3-v)]][1],
                        Teapot::cpdata[Teapot::patchdata[patchNum][u*4+(3-v)]][2]
                        );
            } else {
                patch[u][v] = glm::vec3(
                        Teapot::cpdata[Teapot::patchdata[patchNum][u*4+v]][0],
                        Teapot::cpdata[Teapot::patchdata[patchNum][u*4+v]][1],
                        Teapot::cpdata[Teapot::patchdata[patchNum][u*4+v]][2]
                        );
            }
        }
    }
}
void buildPatch(glm::vec3 patch[][4],
                           float *v, int &index, glm::mat3 reflect )
{
	using namespace glm;
    for( int i = 0; i < 4; i++ )
    {
        for( int j = 0 ; j < 4; j++)
        {
            glm::vec3 pt = reflect * patch[i][j];

            v[index] = pt.x;
            v[index+1] = pt.y;
            v[index+2] = pt.z;

            index += 3;
        }
    }
}
void buildPatchReflect(int patchNum,
           float *v, int &index, bool reflectX, bool reflectY)
{
	
    glm::vec3 patch[4][4];
    glm::vec3 patchRevV[4][4];
    getPatch(patchNum, patch, false);
    getPatch(patchNum, patchRevV, true);

    // Patch without modification
    buildPatch(patchRevV, v, index, glm::mat3(1.0f));

    // Patch reflected in x
    if( reflectX ) {
        buildPatch(patch, v,
                   index, glm::mat3(glm::vec3(-1.0f, 0.0f, 0.0f),
                               glm::vec3(0.0f, 1.0f, 0.0f),
                               glm::vec3(0.0f, 0.0f, 1.0f) ) );
    }

    // Patch reflected in y
    if( reflectY ) {
        buildPatch(patch, v,
                   index, glm::mat3(glm::vec3(1.0f, 0.0f, 0.0f),
                               glm::vec3(0.0f, -1.0f, 0.0f),
                               glm::vec3(0.0f, 0.0f, 1.0f) ) );
    }

    // Patch reflected in x and y
    if( reflectX && reflectY ) {
        buildPatch(patchRevV, v,
                   index, glm::mat3(glm::vec3(-1.0f, 0.0f, 0.0f),
                               glm::vec3(0.0f, -1.0f, 0.0f),
                               glm::vec3(0.0f, 0.0f, 1.0f) ) );
    }
}

void generatePatches(float * v) {
    int idx = 0;

    // Build each patch
    // The rim
    buildPatchReflect(0, v, idx, true, true);
    // The body
    buildPatchReflect(1, v, idx, true, true);
    buildPatchReflect(2, v, idx, true, true);
    // The lid
    buildPatchReflect(3, v, idx, true, true);
    buildPatchReflect(4, v, idx, true, true);
    // The bottom
    buildPatchReflect(5, v, idx, true, true);
    // The handle
    buildPatchReflect(6, v, idx, false, true);
    buildPatchReflect(7, v, idx, false, true);
    // The spout
    buildPatchReflect(8, v, idx, false, true);
    buildPatchReflect(9, v, idx, false, true);
}
void InitTeapotVertexArray()
{
	int verts = 32 * 16;
    float * v = new float[ verts * 3 ];

	generatePatches(v);
	
	teapotVA = new  IGLUVertexArray();
	   
	 // Set up a list of vertices 
	 teapotVA->SetVertexArray( verts * 3 * sizeof(float),     // Array size of vertex data (in bytes)
		                       v );             // Array with per-vertex data inside
	 teapotVA->EnableAttribute( shader[2]["VertexPosition"],   // What GLSL vertex attribute is getting data?
		                        3, GL_FLOAT,        // How many bits of data will that variable get? (2 floats)
								0,                  // What is the stride between data for adjacent vertices?
								BUFFER_OFFSET(0) ); // Where in the array is the first vertex located (in bytes from start)
	 delete[] v;
}

void OpenGLInitialization()
{
	// Standard OpenGL setup
	glewInit();	

	// Create our frame timer
	frameTime = new IGLUFrameRate( 20 );

	// Load the texture with IGLU
	printf("(+) Loading object...\n" );
	float vertexData[]={0,0,0,1};
	vertArray = new IGLUVertexArray();
	
	// Set up a list of vertices 
	vertArray->SetVertexArray( sizeof( vertexData ),     // Array size of vertex data (in bytes)
		                       vertexData );             // Array with per-vertex data inside


	

	// Create a virtual trackball
	ball = new IGLUTrackball( 512, 512 );

	// Load a shader for the object
	printf("(+) Loading shaders...\n" );
	shader[0] = new IGLUShaderProgram( "shaders/passthrough_vertex.glsl","shaders/plane_control.glsl","shaders/plane_tessellation.glsl", "shaders/simple_fragment.glsl" );	
	shader[1] = new IGLUShaderProgram("shaders/passthrough_vertex.glsl","shaders/plane_control.glsl","shaders/sphere_tessellation.glsl", "shaders/simple_fragment.glsl"  );
	
	shader[2] = new IGLUShaderProgram("shaders/tessteapot.vs.glsl","shaders/tessteapot.tcs.glsl","shaders/tessteapot.tes.glsl","shaders/tessteapot.gs.glsl","shaders/tessteapot.fs.glsl");
	GLuint handle = shader[2]->GetProgramID();
	
	int result = glGetUniformLocation( handle, "LineWidth" );
	shader[2]["LightPosition"] = iglu::vec4(0.f,0.f,0.0f,-1.0f);
	shader[2]["LightIntensity"] = iglu::vec3(1.0f,1.0f,1.0f);
	shader[2]["Kd"] = iglu::vec3(0.9f, 0.9f, 1.0f);
	shader[2]["LineWidth"] = .8f;
	shader[2]["LineColor"] = vec4(0.05f,0.0f,0.05f,1.0f);
	// Start the GLUT interaction loop

	// OK.  Now set this array up to use in drawing.  
	//   NOTE:  If multiple shaders are to be used, this may need to be redone when 
	//          the shader used is changed.  (Though it may not be, if you understand the 
	//          underlying OpenGL & GLSL compilation and linking process -- all that's needed
	//          to be correct is the attribute index, which is computed from shader["vertex"])
	vertArray->EnableAttribute( shader[0]["vertex"],   // What GLSL vertex attribute is getting data?
		                        4, GL_FLOAT,        // How many bits of data will that variable get? (2 floats)
								0,                  // What is the stride between data for adjacent vertices?
								BUFFER_OFFSET(0) ); // Where in the array is the first vertex located (in bytes from start)
	vertArray->EnableAttribute( shader[1]["vertex"],   // What GLSL vertex attribute is getting data?
		                        4, GL_FLOAT,        // How many bits of data will that variable get? (2 floats)
								0,                  // What is the stride between data for adjacent vertices?
								BUFFER_OFFSET(0) ); // Where in the array is the first vertex located (in bytes from start)

	InitTeapotVertexArray();
}
int main(int argc, char** argv)
{
	// Put together our main window
	myWin = new IGLUWindow( 512, 512, "Simple Tessellation Shader Demo" );
	myWin->SetWindowProperties( IGLU_WINDOW_NO_RESIZE |	
								IGLU_WINDOW_DOUBLE |
								IGLU_WINDOW_REDRAW_ON_IDLE |
								IGLU_WINDOW_W_FRAMERATE ); 
	myWin->SetDisplayCallback( display );  
	myWin->SetKeyboardCallback(keys);
	myWin->SetMouseButtonCallback(mouse);
	myWin->SetActiveMotionCallback(motion);	
	myWin->SetIdleCallback( IGLUWindow::NullIdle );
	myWin->SetPreprocessOnGLInit( OpenGLInitialization );
	myWin->CreateWindow( argc, argv );     
	
	// Create a widget window to allow us to interact/change our IGLUVariables
	uiWin = new IGLUWidgetWindow( 300, 220, "UI Widget Window" );
	uiWin->AddWidget( &innerTessFactor );
	uiWin->AddWidget( &outerTessFactor );
	uiWin->AddWidget( &wireframe );	
	uiWin->AddWidget( &light );	
	// Tell our main window about the widget window... 
	myWin->SetWidgetWindow( uiWin );

	
	myWin->Run();
	return 0;
}