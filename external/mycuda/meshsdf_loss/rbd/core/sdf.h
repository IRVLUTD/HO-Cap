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
// Copyright (c) 2013-2016 NVIDIA Corporation. All rights reserved.

#pragma once

#include "core.h"
#include "maths.h"

struct Mesh;

// 2d and 3d signed distance field computation using fast marching method (FMM), output array
// should be the same size as input, non-zero input pixels will have distance < 0.0f, resulting 
// distance is scaled by 1 / max(dimension)
void MakeSDF(const uint32_t* input, uint32_t width, uint32_t height, float* output);
void MakeSDF(const uint32_t* input, uint32_t width, uint32_t height, uint32_t depth, float* output);

// make an SDF from a mesh directly, more accurate than the above
float* MakeSDF(const Mesh* mesh, Vec3 lower, Vec3 upper, uint32_t resolution, bool solid, int& dimx, int& dimy, int& dimz);

// analytic sdf funcs
CUDA_CALLABLE inline float opUnion( float d1, float d2 ) {  return Min(d1,d2); }

CUDA_CALLABLE inline float opSubtraction( float d1, float d2 ) { return Max(-d1,d2); }

CUDA_CALLABLE inline float opIntersection( float d1, float d2 ) { return Max(d1,d2); }
	
CUDA_CALLABLE inline float opExtrusion( Vec3 p, float d, float h )
{
	Vec2 w = Vec2( d, Abs(p.z) - h );
	return Min(Max(w.x,w.y),0.0f) + Length(Max(w,Vec2(0.0f)));
}

CUDA_CALLABLE inline Vec2 opRevolution( Vec3 p, float w )
{
	return Vec2( Length(Vec2(p.x, p.z)) - w, p.y );
}

CUDA_CALLABLE inline float sdRect( Vec2 p, Vec2 b )
{
	Vec2 d = Abs(p)-b;
	return Length(Max(d,Vec2(0.0f))) + Min(Max(d.x,d.y),0.0f);
}

CUDA_CALLABLE inline float sdCircle( Vec2 p, float r )
{
	return Length(p) - r;
}

CUDA_CALLABLE inline float sdBox( Vec3 p, Vec3 b )
{
	Vec3 q = Abs(p) - b;
	return Length(Max(q,Vec3(0.0f))) + Min(Max(q.x,Max(q.y,q.z)),0.0f);
}

CUDA_CALLABLE inline float sdSphere( Vec3 p, float s )
{
	return Length(p)-s;
}

CUDA_CALLABLE inline float sdPie( Vec2 p, Vec2 c, float r)
{
	p.x = abs(p.x);
	float l = Length(p) - r;
	float m = Length(p - c*Clamp(Dot(p,c),0.0f,r) );
	return Max(l,m*Sign(c.y*p.x-c.x*p.y));
}

CUDA_CALLABLE inline float sdSpindle(Vec2 p, float radius, float height, float innerPos, float innerRadius)
{
	float b1 = sdRect(p, Vec2(radius, height));
	//float b2 = sdBox(p - vec2(0.75, 0.0), vec2(outerWidth, length*0.8));
	float b2 = sdCircle(p - Vec2(innerPos, 0.0f), innerRadius);
    
	return opSubtraction(b2, b1);
}

CUDA_CALLABLE inline float sdTest(Vec3 x, float time)
{
	Vec3 p = Vec3(x.z, x.x, x.y);
	return sdSpindle(opRevolution(p, 0.0f), 0.5f, 0.5f, 0.5f, 0.2f - cosf(time)*0.2f) - 0.01f;
}

CUDA_CALLABLE inline Vec3 sdTestGrad(Vec3 x, float time)
{
	float eps = 1.e-2f;

	float dx = sdTest(x + Vec3(eps, 0.0f, 0.0f), time) - sdTest(x - Vec3(eps, 0.0f, 0.0f), time);
	float dy = sdTest(x + Vec3(0.0f, eps, 0.0f), time) - sdTest(x - Vec3(0.0f, eps, 0.0f), time);
	float dz = sdTest(x + Vec3(0.0f, 0.0f, eps), time) - sdTest(x - Vec3(0.0f, 0.0f, eps), time);

	return SafeNormalize(Vec3(dx, dy, dz));
}