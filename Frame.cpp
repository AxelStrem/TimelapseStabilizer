#include <Windows.h>
#include "Frame.h"

//WTF microsoft
#undef min
#undef max
#include <algorithm>
#include <array>

Frame::Frame(int width, int height) : sx(width), sy(height), data(width*height)
{}

Frame::Frame(std::string fpath)
{

}

Frame::~Frame()
{
}

Pix& Frame::GetPixel(int x, int y)
{
	return data[y*sx + x];
}

const Pix& Frame::GetPixel(int x, int y) const
{
	return data[y*sx + x];
}

const Frame& Frame::GetMipmap(int level) const
{
	if (level <= 0) return *this;
	if (!pMip)
	{
		pMip = std::make_unique<Frame>(sx / 2, sy / 2, [this](int x, int y)
		{ return (GetPixel(x * 2, y * 2) + GetPixel(x * 2 + 1, y * 2) + GetPixel(x * 2, y * 2 + 1) + GetPixel(x * 2 + 1, y * 2 + 1))*0.25f; });
	}

	return pMip->GetMipmap(level - 1);
}


double error_estimate(const Frame& base_frame, const FrameScaler& test_frame)
{
	double result = 0.f;
	double total_weight = 0.f;
	float xs = base_frame.GetWidth();
	float ys = base_frame.GetHeight();

	base_frame.traverse([&](int x, int y, const Pix& c1)
	{
		float vx = x / xs - 0.5f;
		float vy = y / xs - 0.5f; // sic
		double dist2 = vx * vx + vy * vy;
		double weight = 1.f;// / (dist2*2000.f);// exp(-dist2 * 2.f);
			//1.f;// / (dist*dist*2000.f);
		if (weight >= 1.f) weight = 1.f;
		Pix c2 = test_frame.GetPixel(x, y);
		if (c2.norm2() > 0)
		{
			result += (c1 - c2).norm2()*weight;
			total_weight+=weight;
		}
	});

	if (total_weight == 0.f) return -1.0;
	return result / total_weight;
}

void fit_transform(const Frame& base_frame, FrameScaler& test_frame, std::priority_queue<transform_params>& candidates, int limit, int variance, int rot_variance, float rot_range)
{
	double best_result = 99999.0;
	float base_rotation = test_frame.rot;
	float base_xshift = test_frame.x_shift;
	float base_yshift = test_frame.y_shift;
	float current_rotation = base_rotation;
	float best_rotation = 0.f;
	float current_xshift = base_xshift;
	float current_yshift = base_yshift;
	float best_xshift = 0.f;
	float best_yshift = 0.f;

	for (int i = 1-variance; i<(variance+1); i++)
		for (int j = 1-variance; j < (variance + 1); j++)
		{
			for (int r = -rot_variance; r <= rot_variance; r++)
			{
				current_xshift = i + base_xshift;
				current_yshift = j + base_yshift;
				current_rotation = (r*rot_range) / (rot_variance) + base_rotation;

				test_frame.SetRotation(current_rotation);
				test_frame.x_shift = current_xshift;
				test_frame.y_shift = current_yshift;

				double current_result = error_estimate(base_frame, test_frame);

				if (current_result < best_result)
				{
					best_result = current_result;
					best_rotation = current_rotation;
					best_xshift = current_xshift;
					best_yshift = current_yshift;
				}
			}

			transform_params tp{ best_xshift, best_yshift, best_rotation, best_result };
			candidates.push(tp);
			best_result = 99999.0;
			while (candidates.size() > limit)
				candidates.pop();
		}

	test_frame.SetRotation(best_rotation);
	test_frame.x_shift = best_xshift;
	test_frame.y_shift = best_yshift;
}

void fit_frame(const Frame & base_frame, const Frame & test_frame, int &xshift, int &yshift, float &rot)
{
	const int iterations = 6;
	
	std::priority_queue<transform_params> pset1;
	
	std::priority_queue<transform_params> pset2;

	pset1.push(transform_params{ 0,0,0,0 });
	
	std::array<int,iterations> step_size{32,16,4,1,1,1};
	std::array<int, iterations> variance{ 16,1,1,1,1,1 };
	std::array<int, iterations> variance_rot{ 16,4,2,1,1,1 };
	std::array<float, iterations> range_rot{ 0.1f,0.05f,0.03f,0.02f,0.01f,0.005f };


	for (int i = 0; i < iterations; i++)
	{
		const Frame& fmm = base_frame.GetMipmap(iterations - 1 - i);
		FrameScaler fs(test_frame.GetMipmap(iterations - 1 - i));
		while (!pset1.empty())
		{
			pset1.top().apply2(fs);
			fit_transform(fmm, fs, pset2, step_size[i], variance[i], variance_rot[i], range_rot[i]);
			pset1.pop();
		}

		std::swap(pset1, pset2);
	}

	xshift = pset1.top().xs;
	yshift = pset1.top().ys;
	rot = pset1.top().rot;
}


void FrameScaler::SetRotation(float rotation)
{
	rot = rotation;
	rot_cos = cos(rotation);
	rot_sin = sin(rotation);
}

Pix FrameScaler::GetPixel(int x, int y) const
{
	float fx = static_cast<float>(x) + x_shift - frame.GetWidth() * 0.5f;
	float fy = static_cast<float>(y) + y_shift - frame.GetHeight() * 0.5f;

	float nx = fx * rot_cos + fy * rot_sin;
	float ny = -fx * rot_sin + fy * rot_cos;

	int rx = nx + frame.GetWidth() * 0.5f;
	int ry = ny + frame.GetHeight() * 0.5f;

	if ((rx < 0) || (rx >= frame.GetWidth()) || (ry < 0) || (ry >= frame.GetHeight()))
		return Pix(0.f, 0.f, 0.f);

	return frame.GetPixel(rx, ry);
}


Pix FrameScaler::GetPixelPrecise(int x, int y) const
{
	float fx = static_cast<float>(x) + x_shift - frame.GetWidth() * 0.5f;
	float fy = static_cast<float>(y) + y_shift - frame.GetHeight() * 0.5f;

	float nx = fx * rot_cos + fy * rot_sin;
	float ny = -fx * rot_sin + fy * rot_cos;

	float rx = nx + frame.GetWidth() * 0.5f;
	float ry = ny + frame.GetHeight() * 0.5f;

	if ((rx < 0) || (rx >= frame.GetWidth()) || (ry < 0) || (ry >= frame.GetHeight()))
		return Pix(0.f, 0.f, 0.f);

	int lx = rx;
	int ly = ry;
	int hx = (lx + 1) % frame.GetWidth();
	int hy = (ly + 1) % frame.GetHeight();

	float wx = rx - lx;
	float wy = ry - ly;

	return (frame.GetPixel(lx, ly)*(1.f-wx)+frame.GetPixel(hx,ly)*wx)*(1.f-wy)+ (frame.GetPixel(lx, hy)*(1.f - wx) + frame.GetPixel(hx, hy)*wx)*(wy);
}