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

	//创建BVH，结果保存在纹理中
	void BuildBvh(iglu::IGLUOBJReader::Ptr* objs, iglu::IGLUMatrix4x4::Ptr matrixes, size_t size);

	//创建虚视锥
	void GenVirtualFrustums(iglu::IGLUOBJReader::Ptr* objs, iglu::IGLUMatrix4x4::Ptr matrixes, size_t size);
}