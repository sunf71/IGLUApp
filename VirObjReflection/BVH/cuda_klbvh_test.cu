#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "cuda_klbvh.h"
#include "MortonCode.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
 #include <thrust/count.h>
#include <thrust/execution_policy.h>
#include "cuda_klbvh.cuh"
#include <algorithm>
#include <cstdlib>

#include "gputimer.cuh"
#include "timer.h"
using namespace nih;

NIH_DEVICE bool AABBOverlap(Bbox3f& boxA, Bbox3f& boxB)
{
	for (int i=0; i<3; i++)
	{
		if (fabs(boxB.m_max[i]+boxB.m_min[i]-boxA.m_max[i]-boxA.m_min[i]) <
			boxA.m_max[i]-boxA.m_min[i] + boxB.m_max[i] - boxB.m_min[i])
			return true;
	}
	return false;
}


bool NIH_HOST_DEVICE GenerateVirFrustum(uint32 id, const Vector3f& eye,const Vector3f& p1,const Vector3f& p2, const Vector3f& p3, float farD, TriFrustum& frustum)
{
	     //求5个平面方程
		//视锥平面法线指向视锥外
		plane_t pTri(p1,p2,p3);	
		
		float d  = pTri.distance(eye);
		//视点不能位于三角形平面法线那一侧
		if (d<= 0)
			return false;

		//求虚视点
		Vector3f fNormal(pTri.a,pTri.b,pTri.c);
		float dir = dot(eye-p1,fNormal);
		Vector3f vEye = eye-fNormal*2.f*dir;
		
		frustum.id = id;
		frustum.planes[0] = plane_t(vEye,p2,p1);
		frustum.planes[1] = plane_t(vEye,p3,p2);
		frustum.planes[2] = plane_t(vEye,p1,p3);
		frustum.planes[3] =  plane_t(p1,p3,p2);
		frustum.planes[4] = pTri;
		Vector3f c = (p1+p2+p3)*1.f/3.f;
		float cosT = d/euclidean_distance(vEye,c);
		frustum.planes[4].d -= farD*cosT;		
		return true;
}

//虚视锥生成kernel
__global__ void GenerateVirFrustumKernel(Vector3f* eye,Vector3f* p123, TriFrustum* frustums,float farD, int count)
{
	uint32 step = blockDim.x * gridDim.x;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
		i < count; 
		i += step) 
	{
		GenerateVirFrustum(i,*eye,p123[i*3],p123[i*3+1],p123[i*3+2],farD,frustums[i]);
		
	}
}

__global__ void GenerateVirtualFrustumKernel(Vector3f* eye,Vector3f* p123, uint32* marixId, float* matrix,TriFrustum* frustums,unsigned* Ids, float farD, int count)
	{
		uint32 step = blockDim.x * gridDim.x;
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
			i < count; 
			i += step) 
		{
			float* mat = &matrix[16*marixId[i]];
			Vector3f p0 = MatrixXVector3f(mat,p123[i*3]);
			Vector3f p1 = MatrixXVector3f(mat,p123[i*3+1]);
			Vector3f p2 = MatrixXVector3f(mat,p123[i*3+2]);
			if (GenerateVirFrustum(i,*eye,p0,p1,p2,farD,frustums[i]))
				Ids[i] = i;
			else
				Ids[i] = unsigned(-1);
		}
	}


bool BboxCompare(const Bbox3f& lbox, const Bbox3f& rbox)
{
	const double zero = 0.0001;

	return (abs(lbox.m_min[0]-rbox.m_min[0])<zero &&
		abs(lbox.m_min[1]-rbox.m_min[1])<zero && 
		abs(lbox.m_min[2]-rbox.m_min[2])<zero &&
		abs(lbox.m_max[0]-rbox.m_max[0])<zero &&
		abs(lbox.m_max[1]-rbox.m_max[1])<zero && 
		abs(lbox.m_max[2]-rbox.m_max[2])<zero );
}


