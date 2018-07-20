#pragma once
#include <windows.h>
#include <vector>
#include <string>
#include <memory>
#include <queue>

struct alignas(16) Pix
{
	float r;
	float g;
	float b;
	float w;

	Pix() : r(0.f), g(0.f), b(0.f), w(1.f) {}
	Pix(COLORREF col) : r(GetRValue(col)/255.f), 
						g(GetGValue(col)/255.f),
						b(GetBValue(col)/255.f),
						w(1.f) {}

	Pix(float r_, float g_, float b_) : r(r_), g(g_), b(b_), w(1.f) {}

	operator COLORREF() const
	{
		return RGB(static_cast<int>(r * 255.f) & 0xFF,
				   static_cast<int>(g * 255.f) & 0xFF,
				   static_cast<int>(b * 255.f) & 0xFF);
	}

	float norm2() const
	{
		return r * r + g * g + b * b;
	}

	BYTE RByte() const { return static_cast<int>(r * 255) & 0xFF; }
	BYTE GByte() const { return static_cast<int>(g * 255) & 0xFF; }
	BYTE BByte() const { return static_cast<int>(b * 255) & 0xFF; }

};

inline Pix operator+(const Pix& a, const Pix& b)
{
	return Pix{ a.r + b.r,a.g + b.g,a.b + b.b };
}

inline Pix operator-(const Pix& a, const Pix& b)
{
	return Pix{ a.r - b.r,a.g - b.g,a.b - b.b };
}

inline Pix& operator+=(Pix& a, const Pix& b)
{
	a.r += b.r, a.g += b.g, a.b += b.b;
	return a;
}

inline Pix& operator-=(Pix& a, const Pix& b)
{
	a.r -= b.r, a.g -= b.g, a.b -= b.b;
	return a;
}

inline Pix operator*(const Pix& a, float f)
{
	return Pix{ a.r * f,a.g * f,a.b * f };
}

inline Pix operator*(float f, const Pix& a)
{
	return a * f;
}

inline Pix operator/(const Pix& a, float f)
{
	return Pix{ a.r / f,a.g / f,a.b / f };
}

inline Pix& operator*=(Pix& a, float f)
{
	return a = a*f;
}

inline Pix& operator/=(Pix& a, float f)
{
	return a = a / f;
}

class Frame
{
	std::vector<Pix> data;
	int sx, sy;
	mutable std::unique_ptr<Frame> pMip;

	std::vector<Pix> blur;

	std::vector<Pix> grad_x;
	std::vector<Pix> grad_y;
public:
	Frame(int width, int height);

	template<class F> Frame(int width, int height, F pix_loader) : Frame(width, height)
	{
		for (int j = 0; j < sy; ++j)
			for (int i = 0; i < sx; ++i)
				data[j*sx + i] = pix_loader(i, j);
	}

	Frame() {}
	Frame(std::string filepath);
	Frame(Frame&& f) : data(std::move(f.data)), sx(f.sx), sy(f.sy), pMip(std::move(f.pMip))
	{}

	Frame& operator=(Frame&& f)
	{
		data = std::move(f.data);
		sx = f.sx;
		sy = f.sy;
		pMip = std::move(f.pMip);
		return *this;
	}
		
	void Clear()
	{
		data.clear();
		pMip.reset();
		sx = sy = 0;
	}

	~Frame();

	void compute_gradient()
	{
		blur = data;
		std::vector<Pix> tmp(data.size());

		for (int k = 0; k < 20; ++k)
		{
			for (int j = 1; j < sy - 1; ++j)
				for (int i = 1; i < sx - 1; ++i)
				{
					tmp[j*sx + i] = (2.f*blur[j*sx + i] + blur[(j + 1)*sx + i] + blur[(j - 1)*sx + i] + blur[j*sx + i + 1] + blur[j*sx + i - 1] +
						0.7f*blur[(j + 1)*sx + i + 1] + 0.7f*blur[(j + 1)*sx + i - 1] + 0.7f*blur[(j - 1)*sx + i + 1] + 0.7f*blur[(j - 1)*sx + i - 1]) / 8.8f;
				}

			for (int j = 0; j < sy; ++j)
			{
				tmp[j*sx] = blur[j*sx];
				tmp[(j + 1)*sx - 1] = blur[(j + 1)*sx - 1];
			}

			for (int i = 0; i < sx; ++i)
			{
				tmp[i] = blur[i];
				tmp[(sy - 1)*sx + i] = blur[(sy - 1)*sx + i];
			}

			std::swap(tmp, blur);
		}

		grad_x.resize(data.size());
		grad_y.resize(data.size());

		for (int j = 1; j < sy - 1; ++j)
			for (int i = 1; i < sx - 1; ++i)
			{
				grad_x[j*sx + i] = blur[j * sx + i + 1] - blur[j * sx + i - 1];
				grad_y[j*sx + i] = blur[(j + 1)*sx + i] - blur[(j - 1)*sx + i];
			}

	}

	template<class F> void traverse(F pix_processor)
	{
		for (int j = 0; j < sy; ++j)
			for (int i = 0; i < sx; ++i)
				pix_processor(i, j, data[j*sx + i]);
	}

	template<class F> void traverse(F pix_processor) const
	{
		for (int j = 0; j < sy; ++j)
			for (int i = 0; i < sx; ++i)
				pix_processor(i, j, data[j*sx + i]);
	}

	int GetWidth() const { return sx; }
	int GetHeight() const { return sy; }

	Pix& GetPixel(int x, int y);
	const Pix& GetPixel(int x, int y) const;

	const Frame& GetMipmap(int level) const;

	void ClearData()
	{
		data.clear();
		data.shrink_to_fit();
		pMip.reset();
	}

	void CopyToMemory(char* pDest, int stride) const
	{
		for (int j = 0; j < sy; ++j)
		{
			char* pp = pDest;
			for (int i = 0; i < sx; ++i)
			{
				memcpy(pp, &(GetPixel(i, j)), 12);
				pp += 12;
			}
			pDest += stride;
		}
	}

	const char* GetRawData() const
	{
		return reinterpret_cast<const char*>(data.data());
	}

	char* GetRawData()
	{
		return reinterpret_cast<char*>(data.data());
	}
};

struct FrameScaler
{
	const Frame& frame;
	float rot;
	float rot_cos;
	float rot_sin;
	float x_shift;
	float y_shift;

	FrameScaler(const Frame& f) : frame(f), rot_cos(1.f), rot_sin(0.f), x_shift(0.f), y_shift(0.f), rot(0.f) {};

	void SetRotation(float rotation);

	Pix GetPixel(int x, int y) const;
	Pix GetPixelPrecise(int x, int y) const;


	template<class F> void traverse(F pix_processor, int minx = 0, int miny = 0, int maxx = -1, int maxy = -1) const
	{
		if (maxx < 0) maxx = frame.GetWidth();
		if (maxy < 0) maxy = frame.GetHeight();


		for (int j = miny; j < maxy; ++j)
			for (int i = minx; i < maxx; ++i)
				pix_processor(i, j, GetPixel(i,j));
	}

	template<class F> void traverse_precise(F pix_processor, int minx = 0, int miny = 0, int maxx = -1, int maxy = -1) const
	{
		if (maxx < 0) maxx = frame.GetWidth();
		if (maxy < 0) maxy = frame.GetHeight();


		for (int j = miny; j < maxy; ++j)
			for (int i = minx; i < maxx; ++i)
				pix_processor(i, j, GetPixelPrecise(i, j));
	}
};

double error_estimate(const Frame& base_frame, const FrameScaler& test_frame);

struct transform_params
{
	int xs;
	int ys;
	float rot;
	double err;

	void apply(FrameScaler& fs)
	{
		fs.SetRotation(rot);
		fs.x_shift = xs;
		fs.y_shift = ys;
	}

	void apply2(FrameScaler& fs) const
	{
		fs.SetRotation(rot);
		fs.x_shift = xs*2;
		fs.y_shift = ys*2;
	}
};

inline bool operator<(transform_params p1, transform_params p2)
{
	return p1.err < p2.err;
}

void fit_transform(const Frame& base_frame, FrameScaler& test_frame, std::priority_queue<transform_params>& candidates, int limit, int variance, int rot_variance, float rot_range);
void fit_frame(const Frame& base_frame, const Frame& test_frame, int& sx, int& sy, float& rot);

class FrameStream
{
	struct FrameHeader
	{
		Frame frame;
		int relative_xshift;
		int relative_yshift;
		float relative_rotation;

		FrameHeader(Frame&& f, int rxs, int rys, float rr) : frame(std::move(f)), relative_xshift(rxs), relative_yshift(rys), relative_rotation(rr)
		{}
	};

	std::vector<FrameHeader> data;

public:
	void AddFrame(Frame&& f)
	{
		int rsx = 0;
		int rsy = 0;
		float rrot = 0.f;
		if (!data.empty())
		{
			fit_frame(data.back().frame, f, rsx, rsy, rrot);
			data.back().frame.ClearData();

		}
		data.emplace_back(std::move(f), rsx, rsy, rrot);
	}

	template<class F> void traverse_headers(F header_processor) const
	{
		for (int i = 0; i < data.size(); ++i)
			header_processor(i, data[i].relative_xshift, data[i].relative_yshift, data[i].relative_rotation);
	}

	template<class F> void process_last(F header_processor) const
	{
		if (data.empty()) return;
			header_processor(data.back().relative_xshift, data.back().relative_yshift, data.back().relative_rotation, data.back().frame);
	}
};