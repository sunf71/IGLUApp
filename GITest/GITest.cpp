#include "IGLUApp.h"
#include "VFCIGLUApp.h"
#include "GIMApp.h"
#include "TestApp.h"
#include "igluRandom.h"
#include <gl/gl.h>
#include "Sphere.h"
#define NUM_TREES				10000
#define NUM_GRASS				40*40



class GITestApp : public IGLUApp
{
public:
	//GITestApp();
	GITestApp(const char* fileName):IGLUApp(fileName)
	{
		_grassExtent = vec3( 2.3f, 0.3f, 2.3f );
		_skyLight =  vec3(-0.316227766f, 0.948683298f, 0.0f);
		_fogColor = vec3(0.f);
	}
private:
	vec3 _grassExtent, _skyLight, _fogColor;
	float _dtTime;
	IGLUBuffer::Ptr _grassBuffer;
	IGLUTextureBuffer::Ptr _grassTexBuffer;
	GLuint grassBO;
	IGLUBuffer::Ptr _grassUB;
	int _numOfInstance;
	InstanceData * _instanceData;
	// The transform feedback OpenGL ID.  
	IGLUTransformFeedback::Ptr _feedback;
	IGLUBuffer::Ptr _feedbackBuffer;
	IGLUVertexArray::Ptr _instanceVA;
	GLuint _culledGrassQuery;

	void InitTransformFeedback()
	{		
		_instanceVA = new IGLUVertexArray();
		_instanceVA->SetVertexArray(sizeof(InstanceData)*NUM_GRASS,_instanceData);
		_instanceVA->EnableAttribute(_shaders[3]["InstancePosition"],4,GL_FLOAT);
		 _feedback = new IGLUTransformFeedback();
		_feedbackBuffer = new IGLUBuffer();
		_feedbackBuffer->SetBufferData(NUM_GRASS*sizeof(InstanceData));
		_grassTexBuffer = new IGLUTextureBuffer();
		_grassTexBuffer->BindBuffer(GL_RGBA32F,_feedbackBuffer);
		_feedback->AttachBuffer(_feedbackBuffer);
		const char* varyings[] = {"CulledPosition"};
		_shaders[3]->SetTransformFeedbackVaryings(1,varyings);    // What gets output to our transform feedback buffer?
		//tessShader->SetTransformFeedbackVaryings("gl_Position");
		_shaders[3]->SetProgramEnables( IGLU_GLSL_RASTERIZE_DISCARD ); 
		// create query object to retrieve culled primitive count
		glGenQueries(1, &_culledGrassQuery);
	}
	float random(float min,float max)
	{
		IGLURandom::Ptr rand = new IGLURandom();
		float ret =  rand->fRandom()*(max-min) + min;
		delete rand;
		return ret;
		
	}
	void InitGLBuffer()
	{
		glGenBuffers(1, &this->grassBO);
		glBindBuffer(GL_TEXTURE_BUFFER, this->grassBO);
		glBufferData(GL_TEXTURE_BUFFER, sizeof(InstanceData)*NUM_GRASS, NULL, GL_STATIC_DRAW);

		// generate the tree instance data
		InstanceData* instance = (InstanceData*)glMapBufferRange(GL_TEXTURE_BUFFER, 0, sizeof(InstanceData)*NUM_GRASS,
								 GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT | GL_MAP_UNSYNCHRONIZED_BIT);
		int size = (int)sqrt((float)NUM_GRASS);
		for (int i=0; i<size; i++) {
			for (int j=0; j<size; j++) {
				instance[i+j*size].position = vec4( (i-size/2)*4.0f+random(-0.05,0.05), -5.5f, (j-size/2)*4.0f+random(-0.05,0.05), 0.0f );
				//instance[i+j*size].position = vec4(1.f);
			}
		}
		glUnmapBuffer(GL_TEXTURE_BUFFER);
	}
	void InitIGLUBuffer()
	{
		_grassBuffer = new IGLUBuffer();				
		_grassBuffer->SetBufferData(sizeof(InstanceData)*NUM_GRASS,_instanceData);		
		
	}
	void InitUniformBuffer()
	{
		_grassUB = new IGLUBuffer(IGLU_UNIFORM);		

		// Copy the data into the buffer
		_grassUB->SetBufferData( NUM_GRASS*sizeof(InstanceData), _instanceData, iglu::IGLU_STATIC | iglu::IGLU_DRAW );
		_grassUB->Unbind();
	}
	void InitTexture()
	{
		_grassBuffer = new IGLUBuffer();			
		_grassBuffer->SetBufferData(sizeof(InstanceData)*NUM_GRASS,_instanceData);			
		_grassTexBuffer = new IGLUTextureBuffer();
		_grassTexBuffer->BindBuffer(GL_RGBA32F,_grassBuffer);
	}
	void InitInstanceData()
	{
		_instanceData = new InstanceData[NUM_GRASS];		
		int size = (int)sqrt((float)NUM_GRASS);
		for (int i=0; i<size; i++) {
			for (int j=0; j<size; j++) {
				_instanceData[i+j*size].position = vec4( (i-size/2)*4.0f+random(-0.05,0.05), -5.5f, (j-size/2)*4.0f+random(-0.05,0.05), 0.0f );
				_instanceData[i+j*size].normal = vec4(1,0,0,0);
				}
		}
		InitIGLUBuffer();
		//InitTexture();
		//InitUniformBuffer();
		//InitTransformFeedback();
	
	}

public:
	virtual void InitScene()
	{
		IGLUApp::InitScene();
		
		InitInstanceData();		
		
		_dtTime = 0.f;
	}
	void UpdateBuffer()
	{
		_grassBuffer->SetBufferData(sizeof(InstanceData)*NUM_GRASS,_instanceData);		
	}
	virtual void InitShaders()
	{
		IGLUShaderProgram::Ptr giShader = new IGLUShaderProgram("../../CommonSampleFiles/shaders/normal.vs","../../CommonSampleFiles/shaders/normal.fs");
		_shaders.push_back(giShader);
		IGLUShaderProgram::Ptr shader = new IGLUShaderProgram("../../CommonSampleFiles/shaders/normalT.vs","../../CommonSampleFiles/shaders/normalT.fs");
		_shaders.push_back(shader);
		IGLUShaderProgram::Ptr Ushader = new IGLUShaderProgram("../../CommonSampleFiles/shaders/normalU.vs","../../CommonSampleFiles/shaders/normalU.fs");
		_shaders.push_back(Ushader);

		_shaders.push_back(new IGLUShaderProgram("../../CommonSampleFiles/shaders/cull.vs","../../CommonSampleFiles/shaders/cull.gs",NULL));
	}
	virtual ~GITestApp()
	{
		delete _grassBuffer;
		delete _grassTexBuffer;
		delete[] _instanceData;
		delete _feedback;
		delete _feedbackBuffer;
	}
	void DisplayTexture()
	{
		_dtTime += 0.001;		
		_shaders[1]->Enable();

		_shaders[1]["ModelViewMatrix"] = _camera->GetViewMatrix();;
		_shaders[1]["ProjectionMatrix"] = _camera->GetProjectionMatrix();
		_shaders[1]["TimeFactor"] = _dtTime;
		_shaders[1]["SkyLightDir"] = vec3(-0.316227766f, 0.948683298f, 0.0f);
		_shaders[1]["FogColor"] = vec3(0.f);
		_shaders[1]["matlInfoTex"]     = IGLUOBJMaterialReader::s_matlCoefBuf;
		_shaders[1]["matlTextures"]    = IGLUOBJMaterialReader::s_matlTexArray;
		//_shaders[1]["InstanceData"] = _grassTexBuffer;
		_objReaders[0]->DrawMultipleInstances(_shaders[1],_grassTexBuffer,NUM_GRASS);
		//_objReaders[0]->Draw(_shaders[1]);
	
		_shaders[1]->Disable();

	}
	void DisplayVA()
	{
		_dtTime += 0.1;
		/*UpdateBuffer();*/
		_shaders[0]->Enable();

		_shaders[0]["ModelViewMatrix"] = _camera->GetViewMatrix();;
		_shaders[0]["ProjectionMatrix"] = _camera->GetProjectionMatrix();
		_shaders[0]["TimeFactor"] = _dtTime;
		_shaders[0]["SkyLightDir"] = vec3(-0.316227766f, 0.948683298f, 0.0f);
		_shaders[0]["FogColor"] = vec3(0.f);
		_shaders[0]["matlInfoTex"]     = IGLUOBJMaterialReader::s_matlCoefBuf;
		_shaders[0]["matlTextures"]    = IGLUOBJMaterialReader::s_matlTexArray;
		_objReaders[0]->DrawMultipleInstances(_shaders[0],_grassBuffer,NUM_GRASS);
		//_objReaders[0]->Draw(_shaders[1]);
	
		_shaders[0]->Disable();
	}
	void DisplayUniform()
	{
		_dtTime += 0.001;		
		_shaders[2]->Enable();

		_shaders[2]["ModelViewMatrix"] = _camera->GetViewMatrix();;
		_shaders[2]["ProjectionMatrix"] = _camera->GetProjectionMatrix();
		_shaders[2]["TimeFactor"] = _dtTime;
		_shaders[2]["SkyLightDir"] = _skyLight;
		_shaders[2]["FogColor"] = _fogColor;
		_shaders[2]["matlInfoTex"]     = IGLUOBJMaterialReader::s_matlCoefBuf;
		_shaders[2]["matlTextures"]    = IGLUOBJMaterialReader::s_matlTexArray;
		// Find the bind point for this uniform
		int uniformBlkIdx = glGetUniformBlockIndex( _shaders[2]->GetProgramID(), "InstanceData" );
		 glBindBufferBase( GL_UNIFORM_BUFFER, uniformBlkIdx, _grassUB->GetBufferID() );
		_objReaders[0]->DrawMultipleInstances(_shaders[2], NUM_GRASS);
		_shaders[2]->Disable();
	}
	void DisplayGSCull()
	{
		//��һ����ƣ�tranform feedback ��׶��ü�
		glEnable(GL_RASTERIZER_DISCARD); 
		_shaders[3]->Enable();

		_shaders[3]["ModelViewMatrix"] = _camera->GetViewMatrix();
		_shaders[3]["ProjectionMatrix"] = _camera->GetProjectionMatrix();
		_shaders[3]["ObjectExtent"] = _grassExtent;

		_feedback->Begin(GL_POINTS);
		// query for generated primitive count
		glBeginQuery(GL_PRIMITIVES_GENERATED, this->_culledGrassQuery);

		_instanceVA->DrawArrays(GL_POINTS,0,NUM_GRASS);

		glEndQuery(GL_PRIMITIVES_GENERATED);

		_feedback->End();
		_shaders[3]->Disable();

		// get the result of the previous query about the generated primitive count
		int visibleGrasses;
		glGetQueryObjectiv(this->_culledGrassQuery, GL_QUERY_RESULT, &visibleGrasses);

		//printf("%d\n",visibleGrasses);
		glDisable(GL_RASTERIZER_DISCARD);
		//�ڶ������
		_dtTime += 0.01;		
		_shaders[1]->Enable();

		_shaders[1]["ModelViewMatrix"] = _camera->GetViewMatrix();
		_shaders[1]["ProjectionMatrix"] = _camera->GetProjectionMatrix();
		_shaders[1]["TimeFactor"] = _dtTime;
		_shaders[1]["SkyLightDir"] = _skyLight;
		_shaders[1]["FogColor"] = _fogColor;
		_shaders[1]["matlInfoTex"]     = IGLUOBJMaterialReader::s_matlCoefBuf;
		_shaders[1]["matlTextures"]    = IGLUOBJMaterialReader::s_matlTexArray;
		//_shaders[1]["InstanceData"] = _grassTexBuffer;
		_objReaders[0]->DrawMultipleInstances(_shaders[1],_grassTexBuffer,visibleGrasses);
		//_objReaders[0]->Draw(_shaders[1]);
	
		_shaders[1]->Disable();

	}
	virtual void Display()
	{
		// Start timing this frame draw
		_frameRate->StartFrame();
	
		// Clear the screen
		glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
	
		//DisplayUniform();
		//DisplayTexture();
		DisplayVA();
		//DisplayGSCull();
		// Draw the framerate on the screen
		char buf[32];
		sprintf( buf, "%.1f fps", _frameRate->EndFrame() );
		IGLUDraw::DrawText( IGLU_FONT_VARIABLE, 0, 0, buf );
	}
};

void Test()
{
	vector<Object* > objects;
	IGLURandom random;
	for (int i=0; i< 100; i++)
	{		
		
		vec3 p = random.RandomSphereVector();
		objects.push_back(new Triangle(random.RandomSphereVector(),random.RandomSphereVector(),random.RandomSphereVector(),i));
		//objects.push_back(new Sphere(Vector3(p.X(),p.Y(),p.Z()),Vector3(p.X(),p.Y(),p.Z()),Vector3(p.X(),p.Y(),p.Z())));
	}
	int start = 0;
	int end = objects.size();
	BBox bb (objects[start]->getBBox());
	BBox bc( (objects)[start]->getCentroid());
	for(uint32_t p = start+1; p < end; ++p) {
		bb.expandToInclude( objects[p]->getBBox());
		bc.expandToInclude(objects[p]->getCentroid());
	}
	BVH  bvh(&objects);
	Vector3 p1(10.9272740,2.957242,2.3705871);
	Vector3 p2(10.927240,3.1498971,2.0491750);
	Vector3 min = ::min(p1,p2);
}

IGLUApp* app;
void main()
{
	app = new GITestApp("../../CommonSampleFiles/scenes/nature.txt");

	//app= new IGLUApp("../../CommonSampleFiles/scenes/cityIsland.txt");	
	//app= new GIMApp("../../CommonSampleFiles/scenes/cityIsland.txt");	
	//app= new TestApp("../../CommonSampleFiles/scenes/cityIsland.txt");	
	//app = new VFCIGLUApp("sponza.txt");
	//GLint  value;
	//glGetIntegerv(GL_MAX_UNIFORM_BLOCK_SIZE,&value);
	//printf("%d",value);
	app->Run();
	
}