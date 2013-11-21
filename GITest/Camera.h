#ifndef CAMERA_H
#define CAMERA_H
#include "SceneHelper.h"
#include "iglu.h"
using namespace iglu;

class Camera
{
public:
	Camera(CameraData& cd)
	{
		_oAt = _at = vec3(&cd.at[0]);
		_oEye = _eye = vec3(&cd.eye[0]);
		_oUp = _up = vec3(&cd.up[0]);
		_fovy = cd.fovy;
		_nearZ = cd.zNear;
		_farZ = cd.zFar;
		_resX = cd.resX;
		_resY = cd.resY;
		_aspecRatio = _resX / _resY;
		_pm= IGLUMatrix4x4::Perspective(_fovy,_aspecRatio,_nearZ,_farZ);		

		_trackBall = new IGLUTrackball( _resX, _resY );	
	}
	~Camera()
	{
		delete _trackBall;
	}
	IGLUMatrix4x4 GetViewMatrix() 
	{
		_vm = IGLUMatrix4x4::LookAt(_eye,_at,_up);
		return _vm * _trackBall->GetMatrix();
	}

	IGLUMatrix4x4 GetProjectionMatrix() const
	{
		return _pm;
	}

	void ResetView()
	{
		_at = _oAt;
		_eye = _oEye;
		_up = _oUp;
		_trackBall->Reset();
	}
	void MoveForward( float speed )
	{
		vec3 forward( _at-_eye );
		forward.Normalize();
		_eye = _eye + speed*forward;
		_at = _at + speed*forward;
	}

	void MoveBackward( float speed )
	{
		vec3 forward( _at-_eye );
		forward.Normalize();
		_eye = _eye - speed*forward;
		_at = _at - speed*forward;
	}
	void MoveLeft( float speed )
	{
		vec3 forward( _at-_eye );
		forward.Normalize();
		vec3 right( forward.Cross( _up ) );
		right.Normalize();
		_eye = _eye - speed*right;
		_at = _at - speed*right;
	}

	void MoveRight( float speed )
	{
		vec3 forward( _at-_eye );
		forward.Normalize();
		vec3 right( forward.Cross( _up ) );
		right.Normalize();
		_eye = _eye + speed*right;
		_at = _at + speed*right;
	}
	void SetOnClick(int x, int y)
	{
		_trackBall->SetOnClick(x,y);
	}

	void Release()
	{
		_trackBall->Release();
	}

	void UpdateOnMotion(int x, int y)
	{
		_trackBall->UpdateOnMotion(x,y);
	}

	vec3 GetViewDirection()
	{
		return  (_trackBall->GetMatrix() *vec4((_at - _eye),0)).xyz();
	}
	float GetResX()
	{
		return _resX;
	}
	float GetResY()
	{
		return _resY;
	}
	vec3 GetAt()
	{
		return _at;
	}
	vec3 GetEye()
	{
		return _eye;
	}
private:

	vec3 _eye, _at, _up;
	//初始值，用于恢复初始视点
	vec3 _oEye, _oAt, _oUp;

	float _fovy, _nearZ, _farZ, _resX, _resY, _aspecRatio;
	IGLUMatrix4x4 _vm, _pm;
	IGLUTrackball::Ptr _trackBall;
};

#endif