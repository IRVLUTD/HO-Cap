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

#include <cuda_runtime_api.h>
#include "util.h"

void FlexDeviceMallocInternal(void** ptr, size_t size, const char* label, const char* file, int line)
{
	cudaMalloc(ptr, size);

#if USE_MEMTRACK
	TrackAlloc(label, *ptr, int(size));
#endif
}


void FlexDeviceFreeInternal(void* ptr)
{
	cudaFree(ptr);

#if USE_MEMTRACK
	UnTrackAlloc(ptr);
#endif
}


cudaTextureObject_t CreateTexture(const Vec4* buffer, int sizeInBytes)
{
	// create texture object
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeLinear;
	resDesc.res.linear.devPtr = (void*)buffer;
	resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
	resDesc.res.linear.desc.x = 32; // bits per channel
	resDesc.res.linear.desc.y = 32; // bits per channel
	resDesc.res.linear.desc.z = 32; // bits per channel
	resDesc.res.linear.desc.w = 32; // bits per channel
	resDesc.res.linear.sizeInBytes = sizeInBytes;

	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.readMode = cudaReadModeElementType;

	cudaTextureObject_t tex;
	cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);

#if USE_MEMTRACK
	TrackAlloc("CreateTextureFloat4", (void*)tex, sizeInBytes);
#endif

	return tex;
}


void DestroyTexture(cudaTextureObject_t tex)
{
	if (tex)
	{
		cudaDestroyTextureObject(tex);

#if USE_MEMTRACK
		UnTrackAlloc((void*)tex);
#endif

	}
}
