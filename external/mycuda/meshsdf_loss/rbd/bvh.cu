// This code contains NVIDIA Confidential Information and is disclosed to you
// under a form of NVIDIA software license agreement provided separately to you.
//
// Notice
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software and related documentation and
// any modifications thereto. Any use, reproduction, disclosure, or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA Corporation is strictly prohibited.
//
// ALL NVIDIA DESIGN SPECIFICATIONS, CODE ARE PROVIDED "AS IS.". NVIDIA MAKES
// NO WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO
// THE MATERIALS, AND EXPRESSLY DISCLAIMS ALL IMPLIED WARRANTIES OF NONINFRINGEMENT,
// MERCHANTABILITY, AND FITNESS FOR A PARTICULAR PURPOSE.
//
// Information and code furnished is believed to be accurate and reliable.
// However, NVIDIA Corporation assumes no responsibility for the consequences of use of such
// information or for any infringement of patents or other rights of third parties that may
// result from its use. No license is granted by implication or otherwise under any patent
// or patent rights of NVIDIA Corporation. Details are subject to change without notice.
// This code supersedes and replaces all information previously supplied.
// NVIDIA Corporation products are not authorized for use as critical
// components in life support devices or systems without express written approval of
// NVIDIA Corporation.
//
// Copyright (c) 20132017 NVIDIA Corporation. All rights reserved.

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector>
#include "bvh.h"

CUDA_CALLABLE inline int CLZ(int x)
{
	int n;
	if (x == 0) return 32;
	for (n = 0; ((x & 0x80000000) == 0); n++, x <<= 1);
	return n;
}

CUDA_CALLABLE inline uint32_t Part1by2(uint32_t n)
{
	n = (n ^ (n << 16)) & 0xff0000ff;
	n = (n ^ (n <<  8)) & 0x0300f00f;
	n = (n ^ (n <<  4)) & 0x030c30c3;
	n = (n ^ (n <<  2)) & 0x09249249;

	return n;
}

// Takes values in the range [0, 1] and assigns an index based Morton codes of length 3*log2(Dim) bits
template <int Dim>
CUDA_CALLABLE inline uint32_t Morton3(float x, float y, float z)
{
	uint32_t ux = Clamp(int(x*Dim), 0, Dim-1);
	uint32_t uy = Clamp(int(y*Dim), 0, Dim-1);
	uint32_t uz = Clamp(int(z*Dim), 0, Dim-1);

	return (Part1by2(uz) << 2) | (Part1by2(uy) << 1) | Part1by2(ux);
}

CUDA_CALLABLE inline PackedNodeHalf MakeNode(const Vec3& bound, int child, bool leaf)
{
	PackedNodeHalf n;
	n.x = bound.x;
	n.y = bound.y;
	n.z = bound.z;
	n.i = (unsigned int)child;
	n.b = (unsigned int)(leaf?1:0);

	return n;
}

// variation of MakeNode through volatile pointers used in BuildHierarchy
CUDA_CALLABLE inline void MakeNode(volatile PackedNodeHalf* n, const Vec3& bound, int child, bool leaf)
{
	n->x = bound.x;
	n->y = bound.y;
	n->z = bound.z;
	n->i = (unsigned int)child;
	n->b = (unsigned int)(leaf?1:0);
}


//////////////////////////////////////////////////////////////////////

void MedianBVHBuilder::Build(BVH& bvh, const Bounds* items, int n)
{
	memset(&bvh, 0, sizeof(BVH));

	bvh.mMaxNodes = 2*n;

	bvh.mNodeLowers = new PackedNodeHalf[bvh.mMaxNodes];
	bvh.mNodeUppers = new PackedNodeHalf[bvh.mMaxNodes];
	bvh.mNumNodes = 0;

	// root is always in first slot for top down builders
	bvh.mRootNode = 0;

	std::vector<int> indices(n);
	for (int i=0; i < n; ++i)
		indices[i] = i;

	BuildRecursive(bvh, items, &indices[0], 0, n, 0);
}


Bounds MedianBVHBuilder::CalcBounds(const Bounds* bounds, const int* indices, int start, int end)
{
	Bounds u;

	for (int i=start; i < end; ++i)
		u = Union(u, bounds[indices[i]]);

	return u;
}

struct PartitionPredicateMedian
{
	PartitionPredicateMedian(const Bounds* bounds, int a) : bounds(bounds), axis(a) {}

	bool operator()(int a, int b) const
	{
		return bounds[a].GetCenter()[axis] < bounds[b].GetCenter()[axis];
	}

	const Bounds* bounds;
	int axis;
};


int MedianBVHBuilder::PartitionObjectsMedian(const Bounds* bounds, int* indices, int start, int end, Bounds rangeBounds)
{
	assert(end-start >= 2);

	Vec3 edges = rangeBounds.GetEdges();

	int longestAxis = LongestAxis(edges);

	const int k = (start+end)/2;

	std::nth_element(&indices[start], &indices[k], &indices[end], PartitionPredicateMedian(&bounds[0], longestAxis));

	return k;
}

struct PartitionPredictateMidPoint
{
	PartitionPredictateMidPoint(const Bounds* bounds, int a, float m) : bounds(bounds), axis(a), mid(m) {}

	bool operator()(int index) const
	{
		return bounds[index].GetCenter()[axis] <= mid;
	}

	const Bounds* bounds;
	int axis;
	float mid;
};


int MedianBVHBuilder::PartitionObjectsMidPoint(const Bounds* bounds, int* indices, int start, int end, Bounds rangeBounds)
{
	assert(end-start >= 2);

	Vec3 edges = rangeBounds.GetEdges();
	Vec3 center = rangeBounds.GetCenter();

	int longestAxis = LongestAxis(edges);

	float mid = center[longestAxis];


	int* upper = std::partition(indices+start, indices+end, PartitionPredictateMidPoint(&bounds[0], longestAxis, mid));

	int k = upper-indices;

	// if we failed to split items then just split in the middle
	if (k == start || k == end)
		k = (start+end)/2;


	return k;
}

int MedianBVHBuilder::BuildRecursive(BVH& bvh, const Bounds* bounds, int* indices, int start, int end, int depth)
	{
		assert(start < end);

		const int n = end-start;
		const int nodeIndex = bvh.mNumNodes++;

		assert(nodeIndex < bvh.mMaxNodes);

		if (depth > bvh.mMaxDepth)
			bvh.mMaxDepth = depth;

		Bounds b = CalcBounds(bounds, indices, start, end);

		const int kMaxItemsPerLeaf = 1;

		if (n <= kMaxItemsPerLeaf)
		{
			bvh.mNodeLowers[nodeIndex] = MakeNode(b.lower, indices[start], true);
			bvh.mNodeUppers[nodeIndex] = MakeNode(b.upper, indices[start], false);
		}
		else
		{
			//int split = PartitionObjectsMidPoint(bounds, bvh.mIndices, start, end, bvh.mNodeBounds[nodeIndex]);
			int split = PartitionObjectsMedian(bounds, indices, start, end, b);

			int leftChild = BuildRecursive(bvh, bounds, indices, start, split, depth+1);
			int rightChild = BuildRecursive(bvh,bounds, indices, split, end, depth+1);

			bvh.mNodeLowers[nodeIndex] = MakeNode(b.lower, leftChild, false);
			bvh.mNodeUppers[nodeIndex] = MakeNode(b.upper, rightChild, false);
		}

		return nodeIndex;
	}



/////////////////////////////////////////////////////////////////////////////////////////


void LinearBVHBuilderCPU::Build(BVH& bvh, const Bounds* items, int n)
{
	memset(&bvh, 0, sizeof(BVH));

	bvh.mMaxNodes = 2*n;

	bvh.mNodeLowers = new PackedNodeHalf[bvh.mMaxNodes];
	bvh.mNodeUppers = new PackedNodeHalf[bvh.mMaxNodes];
	bvh.mNumNodes = 0;

	// root is always in first slot for top down builders
	bvh.mRootNode = 0;

	std::vector<KeyIndexPair> keys;
	keys.reserve(n);

	Bounds totalBounds;
	for (int i=0; i < n; ++i)
		totalBounds = Union(totalBounds, items[i]);

	// ensure non-zero edge length in all dimensions
	totalBounds.Expand(0.001f);

	Vec3 edges = totalBounds.GetEdges();
	Vec3 invEdges = Vec3(1.0f)/edges;

	for (int i=0; i < n; ++i)
	{
		Vec3 center = items[i].GetCenter();
		Vec3 local = (center-totalBounds.lower)*invEdges;

		KeyIndexPair l;
		l.key = Morton3<1024>(local.x, local.y, local.z);
		l.index = i;

		keys.push_back(l);
	}

	// sort by key
	std::sort(keys.begin(), keys.end());

	BuildRecursive(bvh, &keys[0], items,  0, n, 0);

	printf("Created BVH for %d items with %d nodes, max depth of %d\n", n, bvh.mNumNodes, bvh.mMaxDepth);
}


inline Bounds LinearBVHBuilderCPU::CalcBounds(const Bounds* bounds, const KeyIndexPair* keys, int start, int end)
{
	Bounds u;

	for (int i=start; i < end; ++i)
		u = Union(u, bounds[keys[i].index]);

	return u;
}

inline int LinearBVHBuilderCPU::FindSplit(const KeyIndexPair* pairs, int start, int end)
{
	if (pairs[start].key == pairs[end-1].key)
		return (start+end)/2;

	// find split point between keys, xor here means all bits
	// of the result are zero up until the first differing bit
	int commonPrefix = CLZ(pairs[start].key ^ pairs[end-1].key);

	// use binary search to find the point at which this bit changes
	// from zero to a 1
	const int mask = 1 << (31-commonPrefix);

	while (end-start > 0)
	{
		int index = (start+end)/2;

		if (pairs[index].key&mask)
		{
			end = index;
		}
		else
			start = index+1;
	}

	assert(start == end);

	return start;
}

int LinearBVHBuilderCPU::BuildRecursive(BVH& bvh, const KeyIndexPair* keys, const Bounds* bounds, int start, int end, int depth)
{
	assert(start < end);

	const int n = end-start;
	const int nodeIndex = bvh.mNumNodes++;

	assert(nodeIndex < bvh.mMaxNodes);

	if (depth > bvh.mMaxDepth)
		bvh.mMaxDepth = depth;

	Bounds b = CalcBounds(bounds, keys, start, end);

	const int kMaxItemsPerLeaf = 1;

	if (n <= kMaxItemsPerLeaf)
	{
		bvh.mNodeLowers[nodeIndex] = MakeNode(b.lower, keys[start].index, true);
		bvh.mNodeUppers[nodeIndex] = MakeNode(b.upper, keys[start].index, false);
	}
	else
	{
		int split = FindSplit(keys, start, end);

		int leftChild = BuildRecursive(bvh, keys, bounds, start, split, depth+1);
		int rightChild = BuildRecursive(bvh, keys, bounds, split, end, depth+1);

		bvh.mNodeLowers[nodeIndex] = MakeNode(b.lower, leftChild, false);
		bvh.mNodeUppers[nodeIndex] = MakeNode(b.upper, rightChild, false);
	}

	return nodeIndex;
}


////////////////////////////////////////////////////////

void InitBVH(BVH& bvh)
{
	memset(&bvh, 0, sizeof(bvh));
}

void ResizeBVH(BVH& bvh, int numNodes)
{
	if (numNodes > bvh.mMaxNodes)
	{
		const int numToAlloc = CalculateSlack(numNodes);

		FlexDeviceFree(bvh.mNodeLowers);
		FlexDeviceFree(bvh.mNodeUppers);

		DestroyTexture(bvh.mNodeLowersTex);
		DestroyTexture(bvh.mNodeUppersTex);

		FlexDeviceAlloc(&bvh.mNodeLowers, sizeof(PackedNodeHalf)*numToAlloc);
		FlexDeviceAlloc(&bvh.mNodeUppers, sizeof(PackedNodeHalf)*numToAlloc);

		bvh.mNodeLowersTex = CreateTexture((Vec4*)bvh.mNodeLowers, sizeof(PackedNodeHalf)*numToAlloc);
		bvh.mNodeUppersTex = CreateTexture((Vec4*)bvh.mNodeUppers, sizeof(PackedNodeHalf)*numToAlloc);

		bvh.mMaxNodes = numToAlloc;

		if (!bvh.mRootNode)
			FlexDeviceAlloc(&bvh.mRootNode, sizeof(int));
	}

	bvh.mNumNodes = numNodes;
}

void FreeBVH(BVH& bvh)
{
	FlexDeviceFree(bvh.mNodeLowers); bvh.mNodeLowers = NULL;
	FlexDeviceFree(bvh.mNodeUppers); bvh.mNodeUppers = NULL;

	DestroyTexture(bvh.mNodeLowersTex); bvh.mNodeLowersTex = 0;
	DestroyTexture(bvh.mNodeUppersTex); bvh.mNodeUppersTex = 0;

	FlexDeviceFree(bvh.mRootNode);
}

void FreeBVHHost(BVH& bvh)
{
	delete[] bvh.mNodeLowers;
	delete[] bvh.mNodeUppers;

	bvh.mNodeLowers = 0;
	bvh.mNodeUppers = 0;
	bvh.mMaxNodes = 0;
	bvh.mNumNodes = 0;
}

void CloneBVH(const BVH& hostBVH, BVH& deviceBVH)
{
	ResizeBVH(deviceBVH, hostBVH.mMaxNodes);

	// copy host data to device
	cudaMemcpy(deviceBVH.mNodeLowers, &hostBVH.mNodeLowers[0], sizeof(PackedNodeHalf)*hostBVH.mNumNodes, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceBVH.mNodeUppers, &hostBVH.mNodeUppers[0], sizeof(PackedNodeHalf)*hostBVH.mNumNodes, cudaMemcpyHostToDevice);
	cudaMemset(deviceBVH.mRootNode, 0, sizeof(int));
}
