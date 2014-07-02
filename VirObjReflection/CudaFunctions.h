#include <thrust/device_vector.h>
#include "BVH/vector.h"
#include "BVH/bbox.h"
#include "iglu.h"
namespace cuda
{
	//����GPU��ȡ��������Ķ��㣬���ĺͰ�Χ�м���
	void LoadOBJReader(iglu::IGLUOBJReader::Ptr obj, 
		thrust::device_vector<nih::Vector3f>& points,
		thrust::device_vector<nih::Vector3f>& centers,
		thrust::device_vector<nih::Bbox3f>& Bboxes,
		nih::Bbox3f& gBox);

	//����BVH�����������ȫ�������У���������������
	size_t BuildBvh(iglu::IGLUOBJReader::Ptr* objs, iglu::IGLUMatrix4x4::Ptr matrixes, size_t size);

	//��������׶
	void GenVirtualFrustums(iglu::vec3& eye, float farD, iglu::IGLUOBJReader::Ptr* objs, iglu::IGLUMatrix4x4::Ptr matrixes, size_t size);

	//��������׶���ü�,����������׶�ڵ�����������
	size_t VirtualFrustumCulling(size_t triSize,iglu::vec3& eye, float farD, iglu::IGLUOBJReader::Ptr* objs, iglu::IGLUMatrix4x4::Ptr matrixes, size_t objSize, const unsigned int* inElemBuffer, unsigned int* outElemBuffer);

	void UpdateVirtualObject(unsigned* inPtr, unsigned* outPtr, unsigned size);
}