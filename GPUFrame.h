#pragma once
#include "Frame.h"
#include <vector>
#include <utility>
#include <tuple>
#include <map>

class GPUFrameAllocator
{
	std::multimap<std::pair<int, int>, std::pair<void*, int>> saved;
public:

	std::pair<void*, int> AllocPitch(int xs, int ys);
	void FreePitch(int xs, int ys, void* p, int pitch);

	~GPUFrameAllocator();
};

class GPUFrame
{
	void                    *cudaLinearMemory;
	size_t                  pitch;
	int                     w;
	int                     h;

	static GPUFrameAllocator alloc;
public:
	GPUFrame();
	GPUFrame(int width, int height);
	GPUFrame(const GPUFrame& f);
	GPUFrame(GPUFrame&& f);
	GPUFrame(const Frame& f);

	Frame ExtractFrame() const;

	GPUFrame& operator=(const GPUFrame& f);
	GPUFrame& operator=(GPUFrame&& f);

	~GPUFrame();

	GPUFrame blur() const;
	GPUFrame blur_n(int n) const;
	GPUFrame blur_progressive(int step) const;

	GPUFrame transform(float angle, float dx, float dy, float zoom) const;
	GPUFrame compare(const GPUFrame& f) const;

	GPUFrame operator*(const GPUFrame& f) const;

	GPUFrame produce_frame() const;

	std::pair<GPUFrame, GPUFrame>		gradient() const;
	GPUFrame							compute_force(const GPUFrame& original, const std::pair<GPUFrame, GPUFrame>& grad) const;
	GPUFrame							apply_force(float angle, float dx, float dy) const;
	std::tuple<float,float,float,float> reduce_force() const;

	void* GetRawData() const;
};

