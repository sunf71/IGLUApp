#include <thrust/device_vector.h>
#include "BVH/vector.h"
#include "BVH/bbox.h"
#include "iglu.h"
namespace cuda
{
	//利用GPU获取场景对象的顶点，中心和包围盒集合
	void LoadOBJReader(iglu::IGLUOBJReader::Ptr obj, 
		thrust::device_vector<nih::Vector3f>& points,
		thrust::device_vector<nih::Vector3f>& centers,
		thrust::device_vector<nih::Bbox3f>& Bboxes,
		nih::Bbox3f& gBox);

	//创建BVH，结果保存在全局纹理中，返回总三角形数
	size_t BuildBvh(iglu::IGLUOBJReader::Ptr* objs, iglu::IGLUMatrix4x4::Ptr matrixes, size_t size);

	//创建虚视锥
	void GenVirtualFrustums(iglu::vec3& eye, float farD, iglu::IGLUOBJReader::Ptr* objs, iglu::IGLUMatrix4x4::Ptr matrixes, size_t size);

	//创建虚视锥并裁剪,返回在虚视锥内的总三角形数
	size_t VirtualFrustumCulling(size_t triSize,iglu::vec3& eye, float farD, iglu::IGLUOBJReader::Ptr* objs, iglu::IGLUMatrix4x4::Ptr matrixes, size_t objSize, const unsigned int* inElemBuffer, unsigned int* outElemBuffer);

	void UpdateVirtualObject(unsigned* inPtr, unsigned* outPtr, unsigned size);
}