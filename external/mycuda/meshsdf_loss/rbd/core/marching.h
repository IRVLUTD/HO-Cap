#pragma once

#if 1

#include <functional>

#include "maths.h"

// Polygonize a voxelmap using marching cubes algorithm

using TriangleCallback = std::function<void(const Vec3& a, const Vec3& b, const Vec3& c)>;
using SdfCallback = std::function<float(const Vec3& x)>;

void Polygonize(int width, int height, int depth, int* volume, const Vec3& minExtents, float sampling, TriangleCallback onTriangle);
void Polygonize(int width, int height, int depth, float* sdf, const Vec3& minExtents, float sampling, float value, TriangleCallback onTriangle);
void Polygonize(int width, int height, int depth, SdfCallback sdf, const Vec3& lower, const Vec3& upper, float value, TriangleCallback onTriangle);

#endif