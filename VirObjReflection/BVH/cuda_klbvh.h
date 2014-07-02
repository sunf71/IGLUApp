#ifndef CUDA_KLBVH_H
#define CUDA_KLBVH_H
#include "types.h"
#include "bvh.h"
#include "bbox.h"
#include <queue>
#include <thrust/device_vector.h>

#include <cub/util_allocator.cuh>
#include <cub/device/device_radix_sort.cuh>


namespace cuda
{
	struct Primitives
	{
		cub::DoubleBuffer<nih::Bbox3f> boxes;
		cub::DoubleBuffer<nih::uint32> indices;
		cub::DoubleBuffer<nih::uint32> codes;
	};

	struct DBVH
	{
		Bvh_Node* nodes;	
		Bbox3f* nodeBoxes;
		Bbox3f* leafBoxes;
		Bvh_Node* leaves;
		NIH_HOST_DEVICE Bvh_Node* getRoot()
		{
			return nodes;
		}
		NIH_HOST_DEVICE Bbox3f getNodeBox(Bvh_Node* node)
		{
			return nodeBoxes[node->id];

		}
		NIH_HOST_DEVICE Bvh_Node* getLChild(Bvh_Node* node)
		{
			return &nodes[node->getChild(0)];
		}
		NIH_HOST_DEVICE Bvh_Node* getRChild(Bvh_Node* node)
		{
			return &nodes[node->getChild(1)];
		}
		NIH_HOST_DEVICE Bvh_Node* getRLeafChild(Bvh_Node* node)
		{
			return &leaves[node->getChild(1)];
		}
		NIH_HOST_DEVICE Bvh_Node* getLLeafChild(Bvh_Node* node)
		{
			return &leaves[node->getChild(0)];
		}
		NIH_HOST_DEVICE Bbox3f getLeafBox(Bvh_Node* node)
		{
			return leafBoxes[node->pid];
		}
	};
	
	inline void DFSBintree(thrust::host_vector<Bvh_Node>* h_nodes,
		thrust::host_vector<Bvh_Node>* h_leaves,
		thrust::host_vector<Bbox3f>* h_nodeBoxes,
		thrust::host_vector<Bbox3f>* h_leafBoxes,
		Bintree_node*  nbvh)
	{
		Bvh_Node* nodes = &(*h_nodes)[0];
		Bvh_Node* leaves = &(*h_leaves)[0];
		Bbox3f* nodeBoxes = &(*h_nodeBoxes)[0];
		Bbox3f* leafBoxes = &(*h_leafBoxes)[0];
		uint32 offset = 0;
		const uint32 stack_size  = 64;
		Bvh_Node* stack[stack_size];
		uint32 top = 0;
		nodes[0].nid = 0;
		Bvh_Node* node = &nodes[0];
		while(top >0 || node != NULL)
		{
			while(node != NULL)
			{
				//cout<<node->id<<endl;
				SetBox(&nbvh[offset],&nodeBoxes[node->id]);
				//nbvh[offset].box = nodeBoxes[node->id];			
				nbvh[offset].leafStart = node->leafStart;
				nbvh[offset].leafEnd = node->leafEnd;
				nbvh[offset].pid = node->id;
				nbvh[offset].lChild = offset+1;
				node->nid = offset;
				offset++;

				stack[top++] = node;
				if (node->l_isleaf)
				{

					nbvh[offset].leafStart = nbvh[offset].leafEnd = nbvh[offset].pid = node->getChild(0);
					SetBox(&nbvh[offset],&leafBoxes[leaves[node->getChild(0)].pid]);

					offset++;
					//cout<<"leaf "<<node->getChild(0)<<endl;
					node = NULL;
				}
				else
				{
					node = &nodes[node->getChild(0)];
				}
			}
			node = stack[--top];
			nbvh[node->nid].RChild = offset;
			if (node->r_isleaf)
			{			
				uint32 pid = leaves[node->getChild(1)].pid;
				SetBox(&nbvh[offset],&leafBoxes[pid]);
				nbvh[offset].leafStart = nbvh[offset].leafEnd = nbvh[offset].pid = node->getChild(1);
				offset++;
				//cout<<"leaf "<<node->getChild(1)<<endl;
				node = NULL;
			}
			else
				node =  &nodes[node->getChild(1)];
		}

	}

	

	

	class KBvh_Builder
	{
	public:
		inline KBvh_Builder(thrust::device_vector<Bvh_Node>& nodes,
			thrust::device_vector<Bvh_Node>& leaves,
			cub::CachingDeviceAllocator& allocator);
		~KBvh_Builder()
		{
			m_allocator->DeviceFree(m_primitives.codes.d_buffers[0]);
			m_allocator->DeviceFree(m_primitives.codes.d_buffers[1]);
			m_allocator->DeviceFree(m_primitives.indices.d_buffers[1]);
			m_allocator->DeviceFree(m_primitives.indices.d_buffers[0]);
			m_allocator->DeviceFree(m_primitives.boxes.d_buffers[1]);
			m_allocator->DeviceFree(d_temp_storage);
		}
		Bbox3f* getNodeBoxes()
		{
			return m_primitives.boxes.d_buffers[1];
		}
		Bbox3f* getLeafBoxes()
		{
			return m_primitives.boxes.d_buffers[0];
		}
		//get sorted indcies
		uint32* getIndices()
		{
			return m_primitives.indices.d_buffers[1];
		}
		template <typename Iterator, typename BboxIterator>
		void build(nih::Bbox3f globalBox, Iterator begin, Iterator end, BboxIterator boxBegin, BboxIterator boxEnd, DBVH* bvh);
	private:
		Primitives m_primitives;
		thrust::device_vector<Bvh_Node>* m_nodes;
		thrust::device_vector<Bvh_Node>* m_leaves;
		cub::CachingDeviceAllocator*  m_allocator; 
		void    *d_temp_storage;
		size_t  temp_storage_bytes;
	};
}
#include "cuda_klbvh_inline.h"
#endif