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
	size_t VirtualFrustumCulling(size_t triSize,iglu::vec3& eye, float farD, float* mirrorPos, iglu::IGLUMatrix4x4::Ptr* mirroTrans, iglu::IGLUMatrix4x4::Ptr matrixes, size_t mirrorTriSize);
	//创建虚视锥并裁剪,返回在虚视锥内的总三角形数
	//@triSize 需要裁剪的三角形总数
	//@mirrorSize 镜面模型总数
	//@eye 视点
	//@farD 远裁剪面
	//@mirrorPos 镜面位置数组，每个镜面是一个三角形，每个三角形三个顶点，每个顶点3个浮点数
	//@mirroMatrixId 镜面矩阵Id数组，每个镜面三角形有一个Id，表示在镜面矩阵中的索引
	//@matrixes 镜面矩阵，一共mirrorSize个矩阵
	//@mirrorTriSize 所有镜面的总三角形数
	//@mEleInPtr 镜面element
	//@mEleOutPtr 更新的镜面element
	size_t VirtualFrustumCulling(size_t triSize, size_t mirrorSize,
		iglu::vec3& eye, float farD, float* mirrorPos, 
		unsigned* mirroMatrixId, iglu::IGLUMatrix4x4::Ptr matrixes, size_t& mirrorTriSize,
		const unsigned* mEleInPtr, unsigned * mEleOutPtr);

	//@vAttrPtr 虚顶点属性
	//@vCmdPtr 虚顶点绘制命令
	//@mElePtr 镜面索引
	//@elePtr 更新镜面索引
	void UpdateVirtualObject(float* vAttrPtr, unsigned* vCmdPtr,unsigned size);

	//更新镜面列表
	void UpdateMirrorElement(const unsigned* mirrorIds, const unsigned* mCElePtr, unsigned* mElePtr, unsigned size);

	//初始化GPU内存
	//@MaxSize 裁剪结果最大数量
	void InitGPUMemory(size_t MaxSize);

	//释放GPU内存
	void ReleaseGPUMemory();
}