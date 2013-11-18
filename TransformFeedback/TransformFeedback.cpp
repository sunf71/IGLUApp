/*****************************************************************************
** This program performs a subdivision into a Sierpinski triangle using a   **
**     geometry shader to divide a triangle into three, output that into a  **
**     GPU buffer using OpenGL's transform feedback, and then pipes that    **
**     output back into the shader to perform another subdivision.          **
**                                                                          **
** This is equivalent to the demo code that has been on my webpage for a    **
**     number of years, only this version uses IGLU.                        **
**                                                                          **
** Chris Wyman (1/13/2012)                                                  **
*****************************************************************************/
//通过transform feedback将tessellation shader细分的曲面保存在buffer里面
//buffer中保存的数据是顶点位置和顶点的法相
#include "iglu.h"
using namespace iglu;

IGLUTrackball::Ptr		ball = 0;

//细分的参数
IGLUFloat innerTessFactor ( 16.f, IGLURange<float>(1.0,64.0), 1.f,  "innerTessFactor:" );
IGLUFloat  outerTessFactor ( 16.f, IGLURange<float>(1.0,64.0), 1.f,  "outerTessFactor:" );
//用于细分的shader和渲染的shader
IGLUShaderProgram::Ptr tessShader, renderShader;

GLuint TransformFeedbackBuffer;
// 输入和输出(transform feedback)
IGLUVertexArray::Ptr inputTriangle, tfBuffer;

IGLUVertexArray::Ptr     vertArray = 0;

// The transform feedback OpenGL ID.  
IGLUTransformFeedback::Ptr feedback;
GLuint TransformFeedbackObject;
IGLUMatrix4x4 view = IGLUMatrix4x4::LookAt(vec3(4.25f * cos(45.f),3.0f,4.25f * sin(45.f)),vec3(0,0,0),vec3(0,1,0));
IGLUMatrix4x4           proj  = IGLUMatrix4x4::Perspective( 60, 1, 3, 20 );
IGLUMatrix4x4           model = IGLUMatrix4x4::Translate( 0, 0, -11 );

void OpenGLInitialization()
{
	// Standard OpenGL setup
	glewInit();	

	// Load shader
	// Load the shader that outputs the red triangle(s) onto the screen
	tessShader= new IGLUShaderProgram("shaders/passthrough_vertex.glsl","shaders/plane_control.glsl","shaders/sphere_tessellation.glsl", "shaders/simple_fragment.glsl"  );
	

	// Load a shader that takes a buffer of triangles and subdivides each into 3 smaller triangles
	renderShader = new IGLUShaderProgram( "shaders/simpleOutput.vert.glsl", 
	                                                                        "shaders/simpleOutput.frag.glsl" );
	const char* varyings[] = {"vertex","normal"};
	tessShader->SetTransformFeedbackVaryings(2,varyings);    // What gets output to our transform feedback buffer?
	//tessShader->SetTransformFeedbackVaryings("gl_Position");
	tessShader->SetProgramEnables( IGLU_GLSL_RASTERIZE_DISCARD ); // When we use this program, don't actually rasterize. 

	tfBuffer = new IGLUVertexArray();
	tfBuffer->SetVertexArray( ( 1 << 15 ) * sizeof(float) * 4, NULL, IGLU_STREAM|IGLU_COPY );
	tfBuffer->EnableAttribute( renderShader["Vertex"], 3, GL_FLOAT, 6*sizeof(GLfloat), BUFFER_OFFSET( 0 ) );
	tfBuffer->EnableAttribute( renderShader["Normal"], 3, GL_FLOAT, 6*sizeof(GLfloat), BUFFER_OFFSET( 3*sizeof(GLfloat) ) );
	// Create an OpenGL transform feedback object  
	feedback = new IGLUTransformFeedback();
	feedback->AttachBuffer( tfBuffer->GetVertexBuffer() );
	
	//Transform feedback output buffer
	//glGenTransformFeedbacks(1, &TransformFeedbackObject);
	//glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, TransformFeedbackObject);
	//glGenBuffers(1, &TransformFeedbackBuffer);
	//glBindBuffer(GL_ARRAY_BUFFER_ARB, TransformFeedbackBuffer);
	//glBufferData(GL_ARRAY_BUFFER_ARB, 100000*4*4, 0, GL_STATIC_DRAW);
	//glBindBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER, 0, TransformFeedbackBuffer, 0, 100000*4*4);
	//glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER,0,tfBuffer->GetVertexBufferID());
	//glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, 0);
	//glBindTransformFeedback(GL_TRANSFORM_FEEDBACK,0);


	// Load the texture with IGLU
	printf("(+) Loading object...\n" );
	float vertexData[]={0,0,0,1};
	vertArray = new IGLUVertexArray();
	
	// Set up a list of vertices 
	vertArray->SetVertexArray( sizeof( vertexData ),     // Array size of vertex data (in bytes)
		                       vertexData );             // Array with per-vertex data inside


	

	// Create a virtual trackball
	ball = new IGLUTrackball( 512, 512 );


	// Start the GLUT interaction loop

	// OK.  Now set this array up to use in drawing.  
	//   NOTE:  If multiple shaders are to be used, this may need to be redone when 
	//          the shader used is changed.  (Though it may not be, if you understand the 
	//          underlying OpenGL & GLSL compilation and linking process -- all that's needed
	//          to be correct is the attribute index, which is computed from shader["vertex"])
	vertArray->EnableAttribute( tessShader["vertex"],   // What GLSL vertex attribute is getting data?
		                        4, GL_FLOAT,        // How many bits of data will that variable get? (2 floats)
								0,                  // What is the stride between data for adjacent vertices?
								BUFFER_OFFSET(0) ); // Where in the array is the first vertex located (in bytes from start)
	

	
}

void display2(void)
{
	//first pass, render vertices to transform feedback buffer
	glEnable(GL_RASTERIZER_DISCARD); 
	tessShader->Enable();
	feedback->Begin(GL_TRIANGLES);
	//glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, TransformFeedbackObject);
	//glBeginTransformFeedback(GL_TRIANGLES);
	tessShader["MVP"]         = proj * model * ball->GetMatrix();
	tessShader["ModelView"]           = model * ball->GetMatrix();		
	tessShader["light" ] = 1;
	tessShader["innerTessFactor"] = innerTessFactor;
	tessShader["outerTessFactor"] = outerTessFactor;
	glPatchParameteri( GL_PATCH_VERTICES, 1);
	vertArray->DrawArrays(GL_PATCHES,0,1);	
	//glEndTransformFeedback();
	feedback->End();
	
	//glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);
	tessShader->Disable();
    glDisable(GL_RASTERIZER_DISCARD);

	//float* p = (float*)	glMapBuffer( GL_ARRAY_BUFFER, GL_READ_ONLY );
	//TransformFeedbackBuffer
//	float* p = (float*)tfBuffer->MapVertexArray();
	//for(int i=0; i<16; i++)
	//{
	//	printf("%f \n",p[i]);
	//}
	//second pass
	//glBindBuffer(GL_ARRAY_BUFFER, tfBuffer->GetVertexBufferID());
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
	glEnable(GL_DEPTH_TEST);
	renderShader->Enable();
	renderShader["Projection"] = proj;
	tfBuffer->DrawTransformFeedback(GL_TRIANGLES,feedback);
	/*glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6*sizeof(GLfloat), 0);	
	glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6*sizeof(GLfloat), (char*)0 + 3*sizeof(GLfloat));	
	glDrawTransformFeedbackStreamInstanced(GL_TRIANGLES,TransformFeedbackObject,0,1);*/
	//glBindBuffer(GL_ARRAY_BUFFER,0);
	renderShader->Disable();
}
// Handle drawing.
void display ( void )
{
	
	//glEnable(GL_RASTERIZER_DISCARD); 
	//transform feedback 渲染到buffer
	tessShader->Enable();	
	feedback->Begin( GL_TRIANGLES );
	tessShader["MVP"]         = proj * model * ball->GetMatrix();
	tessShader["ModelView"]           = model * ball->GetMatrix();		
	tessShader["light" ] = 1;
	tessShader["innerTessFactor"] = innerTessFactor;
	tessShader["outerTessFactor"] = outerTessFactor;
	glPatchParameteri( GL_PATCH_VERTICES, 1);
	vertArray->DrawArrays(GL_PATCHES,0,1);
	feedback->End();
	tessShader->Disable();


	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
	glEnable(GL_DEPTH_TEST);
	//渲染到屏幕
	renderShader->Enable();
	renderShader["Projection"] = proj;
	tfBuffer->DrawTransformFeedback(GL_TRIANGLES,feedback);	
	renderShader->Disable();
}

// Handle keystrokes.
void keys(unsigned char key, int x, int y)
{
	
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
int main(int argc, char** argv)
{
	IGLUWindow *myWin = new IGLUWindow( 768, 768, "Intermediate IGLU Example:  Sierpinski Triangle via Transform Feedback" );
	myWin->SetWindowProperties( IGLU_WINDOW_DOUBLE |
								IGLU_WINDOW_REDRAW_ON_IDLE |
								IGLU_WINDOW_W_FRAMERATE );
	myWin->SetDisplayCallback( display2 );
	myWin->SetIdleCallback( IGLUWindow::NullIdle );
	myWin->SetKeyboardCallback( keys );		
	myWin->SetMouseButtonCallback(mouse);
	myWin->SetActiveMotionCallback(motion);	
	myWin->SetPreprocessOnGLInit( OpenGLInitialization );
	myWin->CreateWindow( argc, argv );
	IGLUWidgetWindow::Ptr   uiWin = 0;
	// Create a widget window to allow us to interact/change our IGLUVariables
	uiWin = new IGLUWidgetWindow( 300, 220, "UI Widget Window" );
	uiWin->AddWidget( &innerTessFactor );
	uiWin->AddWidget( &outerTessFactor );
	// Tell our main window about the widget window... 
	myWin->SetWidgetWindow( uiWin );
	/* start it! */
	IGLUWindow::Run();
	return 0;
}

