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

#pragma once

#include "core/maths.h"

#include "util.h"

#include <algorithm>

struct PackedNodeHalf
{
	float x;
	float y;
	float z;
	unsigned int i : 31;
	unsigned int b : 1;
};

// ensure packed node half fits into 128 bits
#if _WIN32
static_assert(sizeof(PackedNodeHalf) == 16, "error");
#endif

struct BVH
{
	cudaTextureObject_t mNodeLowersTex;
	cudaTextureObject_t mNodeUppersTex;

	// for bottom up builders the root node does not appear in slot 0
	// this is a single int CUDA alloc that holds the index of the root
	int* mRootNode;

	PackedNodeHalf* __restrict__ mNodeLowers;	// stores the lower spatial bound of the node's children, left child stored in i, leaf flag stored in b
	PackedNodeHalf* __restrict__ mNodeUppers;	// stores the upper spatial bound of the node's children, right child stored in i, flag is unused

	int mNumNodes;
	int mMaxNodes;
	int mMaxDepth;
};


//
void InitBVH(BVH& bvh);
void ResizeBVH(BVH& bvh, int numNodes);
void FreeBVH(BVH& bvh);
void FreeBVHHost(BVH& bvh);

void CloneBVH(const BVH& hostBVH, BVH& deviceBVH);

/////////////////////////////////////////////////////////////////////////////////////////////

class MedianBVHBuilder
{
public:

	void Build(BVH& bvh, const Bounds* items, int n);

private:

	Bounds CalcBounds(const Bounds* bounds, const int* indices, int start, int end);

	int PartitionObjectsMedian(const Bounds* bounds, int* indices, int start, int end, Bounds rangeBounds);
	int PartitionObjectsMidPoint(const Bounds* bounds, int* indices, int start, int end, Bounds rangeBounds);

	int BuildRecursive(BVH& bvh, const Bounds* bounds, int* indices, int start, int end, int depth);
};

/////////////////////////////////////////////////////////////////////////////////////////////

class LinearBVHBuilderCPU
{
public:

	void Build(BVH& bvh, const Bounds* items, int n);

private:

	// calculate Morton codes
	struct KeyIndexPair
	{
		uint32_t key;
		int index;

		inline bool operator < (const KeyIndexPair& rhs) const { return key < rhs.key; }
	};

	Bounds CalcBounds(const Bounds* bounds, const KeyIndexPair* keys, int start, int end);
	int FindSplit(const KeyIndexPair* pairs, int start, int end);
	int BuildRecursive(BVH& bvh, const KeyIndexPair* keys, const Bounds* bounds, int start, int end, int depth);

};


#if __CUDACC__

// bounds query
template <typename Func>
__device__ inline void QueryBVH(const BVH& bvh, const Bounds& b, Func& f, int* __restrict__ stack)
{
	if (bvh.mNumNodes == 0)
		return;

	stack[0] = *bvh.mRootNode;
	int count = 1;

	while (count)
	{
		const int nodeIndex = stack[--count];

		// union to allow 128-bit loads
		union { PackedNodeHalf lower; float4 lowerf; };
		union {	PackedNodeHalf upper; float4 upperf; };

		lowerf = tex1Dfetch<float4>(bvh.mNodeLowersTex, nodeIndex);
		upperf = tex1Dfetch<float4>(bvh.mNodeUppersTex, nodeIndex);

		if (Bounds(&lower.x, &upper.x).Overlaps(b))
		{
			const int leftIndex = lower.i;
			const int rightIndex = upper.i;

			if (lower.b)
			{
				f(leftIndex, &stack[count]);
			}
			else
			{
				stack[count++] = leftIndex;
				stack[count++] = rightIndex;
			}
		}
	}
}

// bounds query (with scale)
template <typename Func>
__device__ inline void QueryBVH(const BVH& bvh, const Vec3& bvhScale, const Bounds& b, Func& f, int* __restrict__ stack)
{
	if (bvh.mNumNodes == 0)
		return;

	stack[0] = *bvh.mRootNode;
	int count = 1;

	while (count)
	{
		const int nodeIndex = stack[--count];

		// union to allow 128-bit loads
		union { PackedNodeHalf lower; float4 lowerf; };
		union {	PackedNodeHalf upper; float4 upperf; };

		lowerf = tex1Dfetch<float4>(bvh.mNodeLowersTex, nodeIndex);
		upperf = tex1Dfetch<float4>(bvh.mNodeUppersTex, nodeIndex);

		Bounds nodeBounds(Vec3(&lower.x)*bvhScale, Vec3(&upper.x)*bvhScale);

		if (nodeBounds.Overlaps(b))
		{
			const int leftIndex = lower.i;
			const int rightIndex = upper.i;

			if (lower.b)
			{
				f(leftIndex, &stack[count]);
			}
			else
			{
				stack[count++] = leftIndex;
				stack[count++] = rightIndex;
			}
		}
	}
}


// ray query
template <typename Func>
__device__ inline void QueryBVH(const BVH& bvh, const Vec3& bvhScale, Vec3 start, Vec3 dir, float maxT, float thickness, Func& f, int* stack)
{
	if (bvh.mNumNodes == 0)
		return;

	stack[0] = *bvh.mRootNode;
	int count = 1;

	Vec3 rcpDir = Vec3(1.0f/dir.x, 1.0f/dir.y, 1.0f/dir.z);

	while (count)
	{
		const int nodeIndex = stack[--count];

		// union to allow 128-bit loads
		union { PackedNodeHalf lower; float4 lowerf; };
		union {	PackedNodeHalf upper; float4 upperf; };

		lowerf = tex1Dfetch<float4>(bvh.mNodeLowersTex, nodeIndex);
		upperf = tex1Dfetch<float4>(bvh.mNodeUppersTex, nodeIndex);

		Bounds nodeBounds(Vec3(&lower.x)*bvhScale, Vec3(&upper.x)*bvhScale);
		nodeBounds.Expand(thickness);

		float t;
		bool hit = IntersectRayAABBFast(start, rcpDir, nodeBounds.lower, nodeBounds.upper, t);

		if (hit && t < maxT)
		{
			const int leftIndex = lower.i;
			const int rightIndex = upper.i;

			if (lower.b)
			{
				f(leftIndex, maxT);
			}
			else
			{
				stack[count++] = leftIndex;
				stack[count++] = rightIndex;
			}
		}
	}
}


// closest point query
template <typename Func>
__device__ inline void QueryBVH(const BVH& bvh, Vec3 point, float minDist, Func& f, int* stack)
{
	if (bvh.mNumNodes == 0)
		return;

	stack[0] = *bvh.mRootNode;
	int count = 1;

	while (count)
	{
		const int nodeIndex = stack[--count];

		// union to allow 128-bit loads
		union { PackedNodeHalf lower; float4 lowerf; };
		union {	PackedNodeHalf upper; float4 upperf; };

		lowerf = tex1Dfetch<float4>(bvh.mNodeLowersTex, nodeIndex);
		upperf = tex1Dfetch<float4>(bvh.mNodeUppersTex, nodeIndex);

		Bounds nodeBounds(Vec3(&lower.x), Vec3(&upper.x));

		const float dist = DistanceToAABB(point, nodeBounds.lower, nodeBounds.upper);

		if (dist < minDist)
		{
			const int leftIndex = lower.i;
			const int rightIndex = upper.i;

			if (lower.b)
			{
				f(leftIndex, minDist);
			}
			else
			{
				stack[count++] = leftIndex;
				stack[count++] = rightIndex;
			}
		}
	}
}

#endif // __CUDACC__
