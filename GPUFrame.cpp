#include "GPUFrame.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_d3d11_interop.h>

GPUFrameAllocator GPUFrame::alloc;
// The CUDA kernel launchers that get called
extern "C"
{
	void cuda_texture_2d(void *surface, size_t width, size_t height, size_t pitch, float t);
	void texture_blur(void *source, void *dest, int width, int height, size_t pitch);
	void texture_produce(void *source, void *dest, int width, int height, size_t pitch);
	void texture_blur_progressive(void *source, void *dest, int width, int height, size_t pitch, int step);

	void texture_blur_n(void *source, void *dest, void* temp, int width, int height, size_t pitch, int n);
	void transform(void *source, void *dest, int width, int height, size_t pitch, float angle, float dx, float dy, float zoom);
	void compare(void *source1, void *source2, void *dest, int width, int height, size_t pitch);
	void multiply(void *source1, void *source2, void *dest, int width, int height, size_t pitch);
	
	void compute_gradient(void *source, void *dest_x, void *dest_y, int width, int height, size_t pitch);
	void compute_force(void *source, void *original, void *grad_x, void *grad_y,void *dest, int width, int height, size_t pitch);
	void apply_force(void *source, void* dest, int width, int height, size_t pitch, float angle, float dx, float dy);
	void reduce_force(void *source, void* tmp1, void* tmp2, int width, int height, size_t pitch, int* out_xs, int* out_ys);
}

std::pair<void*, int> GPUFrameAllocator::AllocPitch(int xs, int ys)
{
	auto p = saved.find(std::make_pair(xs, ys));
	if (p != saved.end())
	{
		auto ret = p->second;
		saved.erase(p);
		return ret;
	}
	else
	{
		void* p;
		size_t pitch;
		auto err = cudaMallocPitch(&p, &pitch, xs * sizeof(float) * 4, ys);
		if (err != cudaSuccess)
			throw 0;
		return std::make_pair(p, pitch);
	}
}

void GPUFrameAllocator::FreePitch(int xs, int ys, void * p, int pitch)
{
	saved.insert({ std::make_pair(xs,ys), std::make_pair(p, pitch) });
}

GPUFrameAllocator::~GPUFrameAllocator()
{
	for(auto p : saved)
		cudaFree(p.second.first);
}


GPUFrame::GPUFrame() : cudaLinearMemory(nullptr)
{}

GPUFrame::GPUFrame(int width, int height)
{
	w = width;
	h = height;
	std::tie(cudaLinearMemory, pitch) = alloc.AllocPitch(width, height);
}

GPUFrame::GPUFrame(const GPUFrame & f) : GPUFrame(f.w, f.h)
{
	cudaMemcpy(cudaLinearMemory, f.cudaLinearMemory, pitch*h, ::cudaMemcpyDefault);
}

GPUFrame::GPUFrame(GPUFrame && f)
{
	w = f.w;
	h = f.h;
	pitch = f.pitch;
	cudaLinearMemory = f.cudaLinearMemory;
	f.cudaLinearMemory = nullptr;
}

GPUFrame::GPUFrame(const Frame & f) : GPUFrame(f.GetWidth(), f.GetHeight())
{
	cudaMemcpy(cudaLinearMemory, f.GetRawData(), pitch * h, ::cudaMemcpyDefault);
}

Frame GPUFrame::ExtractFrame() const
{
	Frame result(w,h);
	cudaMemcpy(result.GetRawData(),cudaLinearMemory, pitch * h, ::cudaMemcpyDefault);
	return result;
}


GPUFrame & GPUFrame::operator=(const GPUFrame & f)
{
	if (cudaLinearMemory)
		alloc.FreePitch(w, h, cudaLinearMemory, pitch);

	w = f.w;
	h = f.h;

	std::tie(cudaLinearMemory, pitch) = alloc.AllocPitch(w, h);

	cudaMemcpy(cudaLinearMemory, f.cudaLinearMemory, pitch * h, ::cudaMemcpyDefault);
	return *this;
}

GPUFrame & GPUFrame::operator=(GPUFrame && f)
{
	std::swap(w, f.w);
	std::swap(h, f.h);
	std::swap(pitch, f.pitch);
	std::swap(cudaLinearMemory, f.cudaLinearMemory);

	return *this;
}

GPUFrame::~GPUFrame()
{
	if(cudaLinearMemory)
		alloc.FreePitch(w, h, cudaLinearMemory, pitch);
}

GPUFrame GPUFrame::blur() const
{
	GPUFrame result(w, h);
	texture_blur(cudaLinearMemory, result.cudaLinearMemory, w, h, pitch);
	return result;
}

GPUFrame GPUFrame::produce_frame() const
{
	GPUFrame result(w, h);
	texture_produce(cudaLinearMemory, result.cudaLinearMemory, w, h, pitch);
	return result;
}

GPUFrame GPUFrame::blur_n(int n) const
{
	GPUFrame result(w, h);
	GPUFrame temp(w, h);

	texture_blur_n(cudaLinearMemory, result.cudaLinearMemory, temp.cudaLinearMemory, w, h, pitch, n);
	return result;
}

GPUFrame GPUFrame::blur_progressive(int step) const
{
	GPUFrame result(w, h);
	texture_blur_progressive(cudaLinearMemory, result.cudaLinearMemory, w, h, pitch, step);
	return result;
}

GPUFrame GPUFrame::transform(float angle, float dx, float dy, float zoom) const
{
	GPUFrame result(w, h);
	::transform(cudaLinearMemory, result.cudaLinearMemory, w, h, pitch, angle, dx, dy, zoom);
	return result;
}

GPUFrame GPUFrame::compare(const GPUFrame & f) const
{
	GPUFrame result(w, h);
	::compare(cudaLinearMemory, f.cudaLinearMemory, result.cudaLinearMemory, w, h, pitch);
	return result;
}

GPUFrame GPUFrame::operator*(const GPUFrame & f) const
{
	GPUFrame result(w, h);
	::multiply(cudaLinearMemory, f.cudaLinearMemory, result.cudaLinearMemory, w, h, pitch);
	return result;
}

std::pair<GPUFrame, GPUFrame> GPUFrame::gradient() const
{
	std::pair<GPUFrame, GPUFrame> grad{{ w,h }, { w,h }};
	::compute_gradient(cudaLinearMemory, grad.first.cudaLinearMemory, grad.second.cudaLinearMemory, w, h, pitch);
	return grad;
}

GPUFrame GPUFrame::compute_force(const GPUFrame & original, const std::pair<GPUFrame, GPUFrame>& grad) const
{
	GPUFrame result(w, h);
	::compute_force(cudaLinearMemory, original.cudaLinearMemory, grad.first.cudaLinearMemory, grad.second.cudaLinearMemory, result.cudaLinearMemory, w, h, pitch);
	return result;
}

GPUFrame GPUFrame::apply_force(float angle, float dx, float dy) const
{
	GPUFrame result(w, h);
	::apply_force(cudaLinearMemory, result.cudaLinearMemory, w, h, pitch, angle, dx, dy);
	return result;
}

std::tuple<float, float, float, float> GPUFrame::reduce_force() const
{
	GPUFrame tmp1(w, h), tmp2(w, h);

	std::pair<int, int> rect;

	::reduce_force(cudaLinearMemory, tmp1.cudaLinearMemory, tmp2.cudaLinearMemory, w, h, pitch, &rect.first, &rect.second);
	auto err = cudaGetLastError();
	if (err != ::cudaSuccess)
	{
		throw 0;
	}

	float dx = 0.f;
	float dy = 0.f;
	float da = 0.f;
	float ds = 0.f;

	static std::vector<float> data;
	data.resize(rect.first*rect.second * 4);

	cudaMemcpy2D(data.data(), rect.first * 16, tmp1.cudaLinearMemory, pitch, rect.first*16, rect.second, ::cudaMemcpyDeviceToHost);
	err = cudaGetLastError();
	if (err != ::cudaSuccess)
	{
		throw 0;
	}

	for(int x = 0; x < rect.first; ++x)
		for (int y = 0; y < rect.second; ++y)
		{
			dx += data[(rect.first*y + x) * 4 + 0];
			dy += data[(rect.first*y + x) * 4 + 1];
			da += data[(rect.first*y + x) * 4 + 2];
			ds += data[(rect.first*y + x) * 4 + 3];
		}

	float weight = 1.f / (rect.first*rect.second);

	return std::make_tuple(dx*weight,dy*weight,da*weight,ds*weight);
}

void * GPUFrame::GetRawData() const
{
	return cudaLinearMemory;
}
