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


texture<uint32> bvhTex;
struct bvhTexHelper
{
	static const uint32 nodeSize = 11;
	static const uint32 LChildOf = 6;
	static const uint32 RChildOf = 7;
	static const uint32 pidOf = 8;
	static const uint32 leafStartOf = 9;
	static const uint32 leafEndOf = 10;
	float p[6];
	NIH_DEVICE float* getBbox(uint32 id)
	{
		uint32 offset = id*nodeSize;
		uint32 t[6];
		
		for(int i=0; i<6; i++)
		{
			t[i] = tex1Dfetch(bvhTex,offset+i);
			p[i] = bitsToFloat(t[i]);
		}

	
		return p;
	}

	NIH_DEVICE uint32 getLChild(uint32 id)
	{
		return tex1Dfetch(bvhTex,id*nodeSize+LChildOf);
	}

	NIH_DEVICE uint32 getRChild(uint32 id)
	{
		return tex1Dfetch(bvhTex,id*nodeSize+RChildOf);
	}
	NIH_DEVICE uint32 getPid(uint32 id)
	{
		return tex1Dfetch(bvhTex,id*nodeSize+pidOf);
	}
	NIH_DEVICE uint32 getleafStart(uint32 id)
	{
		return tex1Dfetch(bvhTex,id*nodeSize+leafStartOf);
	}
	NIH_DEVICE uint32 getleafEnd(uint32 id)
	{
		return tex1Dfetch(bvhTex,id*nodeSize+leafEndOf);
	}
	NIH_DEVICE bool isLeaf(uint32 id)
	{
		return getleafStart(id) == getleafEnd(id);
	}
};




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

FORCE_INLINE NIH_DEVICE void FrustumCulling(TriFrustum& frustum, uint32 frustumId,
	uint32 priSize,
	cullingContext* out)
{
	bvhTexHelper helper;
	uint32 offset = priSize*frustumId;
	const uint32 stack_size  = 64;
	uint32 stack[stack_size];
	uint32 top = 0;
	stack[top++] = 0;
	while(top>0)
	{
		uint32 idx = stack[--top];
		//Bintree_node * node = &bvh[idx];
		uint32 RChild = helper.getRChild(idx);
		uint32 LChild = helper.getLChild(idx);
		
		int ret = Intersect(frustum,helper.getBbox(idx));
		if (ret == 2)
		{
			//相交
			
			if(helper.isLeaf(RChild))
			{
				if (Intersect(frustum,helper.getBbox(RChild)))
				{
					uint32 pid = helper.getPid(RChild);
					out[offset+pid].frustumId = frustumId;
					out[offset+pid].triId = pid;					
				}				
			}
			else
				stack[top++] = RChild;

			if (helper.isLeaf(LChild))
			{
				
				if (Intersect(frustum,helper.getBbox(LChild)))
				{
					uint32 pid = helper.getPid(LChild);
					out[offset+pid].frustumId = frustumId;					
					out[offset+pid].triId = pid;
				}				
			}
			else
				stack[top++] = LChild;
		}
		else if (ret == 1)
		{
			//in
			for(int k= helper.getleafStart(idx); k<=helper.getleafEnd(idx);k++)
			{	
				out[offset+k].frustumId = frustumId;
				out[offset+k].triId = k;
			}

		}
	}
}


__global__ void FrustumCullingKernel(TriFrustum* frustumP, int frustum_num, uint32 priSize,cullingContext* list)
{
	uint32 step = blockDim.x * gridDim.x;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
		i < frustum_num; 
		i += step) 
	{
		TriFrustum frustum = frustumP[i];
		FrustumCulling(frustum,i,priSize,list);
		//FrustumCullingT(frustum,i,bvh,priSize,list);
	}
}

void NIH_HOST_DEVICE GenerateVirFrustum(uint32 id, const Vector3f& eye,const Vector3f& p1,const Vector3f& p2, const Vector3f& p3, float farD, TriFrustum& frustum)
{
	     //求5个平面方程
		//视锥平面法线指向视锥外
		plane_t pTri(p1,p2,p3);	
		
		float d  = pTri.distance(eye);
		//视点不能位于三角形平面法线那一侧
		if (d<= 0)
			return;

		//求虚视点
		Vector3f fNormal(pTri.a,pTri.b,pTri.c);
		float dir = dot(eye-p1,fNormal);
		Vector3f vEye = eye-fNormal*2.f*dir;
		
		frustum.id = id;
		frustum.planes[0] = plane_t(eye,p2,p1);
		frustum.planes[1] = plane_t(eye,p3,p2);
		frustum.planes[2] = plane_t(eye,p1,p3);
		frustum.planes[3] =  plane_t(p1,p3,p2);
		frustum.planes[4] = pTri;
		Vector3f c = (p1+p2+p3)*1.f/3.f;
		float cosT = (-d)/euclidean_distance(eye,c);
		frustum.planes[4].d -= farD/cosT;		
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


