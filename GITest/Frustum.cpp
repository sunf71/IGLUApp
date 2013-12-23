#include "Frustum.h"

void Frustum::CreateVAO()
{
	vec3 dir = _at-_eye;
	dir.Normalize();
	vec3 right = dir.Cross(_up);
    float tanHFov = tan(_fovy / 2/180.0*3.1415926);
	float Hnear = tanHFov * _near;
	float Wnear = Hnear * _aspectRatio;
	float Hfar =  tanHFov * _far;
	float  Wfar = Hfar * _aspectRatio;
	vec3 fc = _eye + dir*_far;
	vec3 ftl = fc + (_up * Hfar) - (right * Wfar);
	vec3 ftr = fc + (_up * Hfar) + (right * Wfar);
	vec3 fbl = fc - (_up * Hfar) - (right * Wfar);
	vec3 fbr = fc - (_up * Hfar) + (right * Wfar);

	vec3 nc = _eye + dir * _near ;

	vec3 ntl = nc + (_up * Hnear) - (right * Wnear);
	vec3 ntr = nc + (_up * Hnear) + (right * Wnear);
	vec3 nbl = nc - (_up * Hnear) - (right * Wnear);
	vec3 nbr = nc - (_up * Hnear) + (right * Wnear);

	_vao = new IGLUVertexArray();
	float * points = new float[27];
	int i=0; 
	for(int j=0; j<3; j++)
		points[i++] = _eye[j];
	for(int j=0; j<3; j++)
		points[i++] = ntl[j];
	for(int j=0; j<3; j++)
		points[i++] = ntr[j];
	for(int j=0; j<3; j++)
		points[i++] = nbl[j];
	for(int j=0; j<3; j++)
		points[i++] = nbr[j];
	for(int j=0; j<3; j++)
		points[i++] = ftl[j];
	for(int j=0; j<3; j++)
		points[i++] = ftr[j];
	for(int j=0; j<3; j++)
		points[i++] = fbl[j];
	for(int j=0; j<3; j++)
		points[i++] = fbr[j];

	_vao->SetVertexArray(sizeof(float)*27,points);

	uint elements[] = {0,1, 0,2, 0,3, 0,4, 1,2, 2,4, 4,3, 3,1, 1,5, 2,6, 3,7, 4,8, 5,6, 6,8, 8,7, 7,5};
	//uint elements[] = {1,2, 2,4, 4,3, 3,1};
	_vao->SetElementArray(GL_UNSIGNED_INT,sizeof(elements)*sizeof(uint),elements);
	_vao->EnableAttribute(0,3,GL_FLOAT,3*sizeof(float),0);


	delete[] points;

}

void Frustum::Draw()
{
	if(_vao == NULL)
		CreateVAO();
	_vao->DrawElements(GL_LINES,32);
}