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
	size_t VirtualFrustumCulling(size_t triSize,iglu::vec3& eye, float farD, float* mirrorPos, iglu::IGLUMatrix4x4::Ptr* mirroTrans, iglu::IGLUMatrix4x4::Ptr matrixes, size_t mirrorTriSize);
	//��������׶���ü�,����������׶�ڵ�����������
	//@triSize ��Ҫ�ü�������������
	//@mirrorSize ����ģ������
	//@eye �ӵ�
	//@farD Զ�ü���
	//@mirrorPos ����λ�����飬ÿ��������һ�������Σ�ÿ���������������㣬ÿ������3��������
	//@mirroMatrixId �������Id���飬ÿ��������������һ��Id����ʾ�ھ�������е�����
	//@matrixes �������һ��mirrorSize������
	//@mirrorTriSize ���о��������������
	//@mEleInPtr ����element
	//@mEleOutPtr ���µľ���element
	size_t VirtualFrustumCulling(size_t triSize, size_t mirrorSize,
		iglu::vec3& eye, float farD, float* mirrorPos, 
		unsigned* mirroMatrixId, iglu::IGLUMatrix4x4::Ptr matrixes, size_t& mirrorTriSize,
		const unsigned* mEleInPtr, unsigned * mEleOutPtr);

	//@vAttrPtr �鶥������
	//@vCmdPtr �鶥���������
	//@mElePtr ��������
	//@elePtr ���¾�������
	void UpdateVirtualObject(float* vAttrPtr, unsigned* vCmdPtr,unsigned size);

	//���¾����б�
	void UpdateMirrorElement(const unsigned* mirrorIds, const unsigned* mCElePtr, unsigned* mElePtr, unsigned size);

	//��ʼ��GPU�ڴ�
	//@MaxSize �ü�����������
	void InitGPUMemory(size_t MaxSize);

	//�ͷ�GPU�ڴ�
	void ReleaseGPUMemory();
}