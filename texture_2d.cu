/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <utility>

#define PI 3.1415926536f

texture<float, 2, cudaReadModeElementType> texRef;
/*
 * Paint a 2D texture with a moving red/green hatch pattern on a
 * strobing blue background.  Note that this kernel reads to and
 * writes from the texture, hence why this texture was not mapped
 * as WriteDiscard.
 */
__global__ void cuda_kernel_texture_2d(unsigned char *surface, int width, int height, size_t pitch, float t)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    float *pixel;

    // in the case where, due to quantization into grids, we have
    // more threads than pixels, skip the threads which don't
    // correspond to valid pixels
    if (x >= width || y >= height) return;

    // get a pointer to the pixel at (x,y)
    pixel = (float *)(surface + y*pitch) + 4*x;

	float r = pixel[0];
	float g = pixel[1];
	float b = pixel[2];

	float strength = ((r - g)*(r - g) + (g - b)*(g - b) + (b - r)*(b - r))/2.f;


    // populate it
    float value_x = 0.5f + 0.5f*cos(t + 10.0f*((2.0f*x)/width  - 1.0f));
    float value_y = 0.5f + 0.5f*cos(t + 10.0f*((2.0f*y)/height - 1.0f));

	float alpha = value_x * strength * 0.28f;
	float beta = value_y * strength * 0.28f;

	r = r * 2.f - 1;
	g = g * 2.f - 1;
	b = b * 2.f - 1;


	float rn = cos(beta)*(r*cos(alpha)+g*sin(alpha))-sin(beta)*b;
	float gn = g*cos(alpha)-r*sin(alpha);
	float bn = b*cos(beta)+sin(beta)*(r*cos(alpha) + g * sin(alpha));

	pixel[0] = (rn + 1) / 2.f;
	pixel[1] = (gn + 1) / 2.f;
	pixel[2] = (bn + 1) / 2.f;

	pixel[3] = 1; // alpha

}

extern "C"
void cuda_texture_2d(void *surface, int width, int height, size_t pitch, float t)
{
    dim3 Db = dim3(32, 32);   // block dimensions are fixed to be 256 threads
    dim3 Dg = dim3((width+Db.x-1)/Db.x, (height+Db.y-1)/Db.y);

    cuda_kernel_texture_2d<<<Dg,Db>>>((unsigned char *)surface, width, height, pitch, t);
}

__global__ void kernel_blur(unsigned char *source, unsigned char *dest, int width, int height, size_t pitch, float r, float weight)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	float *pixel_out;

	// in the case where, due to quantization into grids, we have
	// more threads than pixels, skip the threads which don't
	// correspond to valid pixels
	if (x >= (width) || y >= (height)) return;
	
	int xl = max((int)(x - r), 0);
	int xh = min((int)(x + r), width - 1);
	int yl = max((int)(y - r), 0);
	int yh = min((int)(y + r), height - 1);


	// get a pointer to the pixel at (x,y)
	/*float *pixel00 = (float *)(source + (yl) * pitch) + 4 * (xl);
	float *pixel01 = (float *)(source + (yl) * pitch) + 4 * x;
	float *pixel02 = (float *)(source + (yl) * pitch) + 4 * (xh);
	float *pixel10 = (float *)(source + y * pitch) + 4 * (xl);
	float *pixel11 = (float *)(source + y * pitch) + 4 * x;
	float *pixel12 = (float *)(source + y * pitch) + 4 * (xh);
	float *pixel20 = (float *)(source + (yh) * pitch) + 4 * (xl);
	float *pixel21 = (float *)(source + (yh) * pitch) + 4 * x;
	float *pixel22 = (float *)(source + (yh) * pitch) + 4 * (xh);*/
	
	pixel_out = (float *)(dest + y * pitch) + 4 * x;

	float col[3];
	col[0] = col[1] = col[2] = 0.0f;

	float r2 = r * r;
	for(int ix=xl;ix<=xh;ix++)
		for (int iy = yl; iy <= yh; iy++)
		{
			float* pixel_in = (float *)(source + iy * pitch) + 4 * ix;
			float dx = ix - x;
			float dy = iy - y;
			float f = dx * dx + dy * dy;
			if (f > r2) continue;
			float d2 = 1.f / (f + 30.f);
			for (int i = 0; i < 3; i++)
				col[i] += pixel_in[i] * d2 * weight;
		}
	
	for (int i = 0; i < 3; ++i)
	{
		pixel_out[i] = col[i];
		//pixel_out[i] = (pixel00[i] + pixel22[i] + pixel02[i] + pixel20[i])*0.125 + (pixel01[i]+pixel10[i]+pixel21[i]+pixel12[i])*0.125 + pixel11[i]*0.0;
	}
	pixel_out[3] = 1; // alpha

}

__global__ void kernel_blur_simple(unsigned char *source, unsigned char *dest, int width, int height, size_t pitch)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	float *pixel_out;

	// in the case where, due to quantization into grids, we have
	// more threads than pixels, skip the threads which don't
	// correspond to valid pixels
	if (x >= (width) || y >= (height)) return;

	int xl = max((int)(x - 2), 0);
	int xh = min((int)(x + 2), width - 1);
	int yl = max((int)(y - 2), 0);
	int yh = min((int)(y + 2), height - 1);


	// get a pointer to the pixel at (x,y)
	/*float *pixel00 = (float *)(source + (yl) * pitch) + 4 * (xl);
	float *pixel01 = (float *)(source + (yl) * pitch) + 4 * x;
	float *pixel02 = (float *)(source + (yl) * pitch) + 4 * (xh);
	float *pixel10 = (float *)(source + y * pitch) + 4 * (xl);
	float *pixel11 = (float *)(source + y * pitch) + 4 * x;
	float *pixel12 = (float *)(source + y * pitch) + 4 * (xh);
	float *pixel20 = (float *)(source + (yh) * pitch) + 4 * (xl);
	float *pixel21 = (float *)(source + (yh) * pitch) + 4 * x;
	float *pixel22 = (float *)(source + (yh) * pitch) + 4 * (xh);*/

	const float weight[9] = { 1.f, 1.f, 0.75f, 0.5f, 0.5f, 0.25f, 0.f, 0.f, 0.f };

	float total_weight = 0.f;
	pixel_out = (float *)(dest + y * pitch) + 4 * x;

	float col[3];
	col[0] = col[1] = col[2] = 0.0f;

	for (int ix = xl; ix <= xh; ix++)
		for (int iy = yl; iy <= yh; iy++)
		{
			float* pixel_in = (float *)(source + iy * pitch) + 4 * ix;
			int dx = ix - x;
			int dy = iy - y;
			int r2 = dx * dx + dy * dy;

			float w = weight[r2];
			for (int i = 0; i < 3; i++)
				col[i] += pixel_in[i] * w;

			total_weight += w;
		}

	for (int i = 0; i < 3; ++i)
	{
		pixel_out[i] = col[i] / total_weight;
		//pixel_out[i] = (pixel00[i] + pixel22[i] + pixel02[i] + pixel20[i])*0.125 + (pixel01[i]+pixel10[i]+pixel21[i]+pixel12[i])*0.125 + pixel11[i]*0.0;
	}
	pixel_out[3] = 1; // alpha

}

extern "C"
void texture_blur(void *source, void *dest, int width, int height, size_t pitch)
{
	dim3 Db = dim3(32, 32);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((width + Db.x - 1) / Db.x, (height + Db.y - 1) / Db.y);

	kernel_blur_simple << <Dg, Db >> >((unsigned char *)source, (unsigned char *)dest, width, height, pitch);
}

extern "C"
void texture_blur_n(void *source, void *dest, void* temp, int width, int height, size_t pitch, int n)
{
	if (n <= 1)
		return texture_blur(source, dest, width, height, pitch);
	texture_blur(source, temp, width, height, pitch);
	n--;
	void* p;
	while (n)
	{
		texture_blur(temp, dest, width, height, pitch);
		n--;
		p = temp;
		temp = dest;
		dest = p;
	}
}


__global__ void kernel_blur_progressive(unsigned char *source, unsigned char *dest, int width, int height, size_t pitch, int step)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	float *pixel_out;

	// in the case where, due to quantization into grids, we have
	// more threads than pixels, skip the threads which don't
	// correspond to valid pixels
	if (x >= (width) || y >= (height)) return;

	int xl = x - step;
	int xh = x + step;
	int yl = y - step;
	int yh = y + step;

	pixel_out = (float *)(dest + y * pitch) + 4 * x;

	float col[3];
	col[0] = 0.f;
	col[1] = 0.f;
	col[2] = 0.f;

	float w = 0.f;

	for(int ix = xl; ix <= xh; ix+=step)
		for (int iy = yl; iy <= yh; iy += step)
			{
			  if ((ix < width) && (ix >= 0) && (iy < height) && (iy >= 0))
			  {
				  float* pixel_in = (float *)(source + iy * pitch) + 4 * ix;
				  col[0] += pixel_in[0];
				  col[1] += pixel_in[1];
				  col[2] += pixel_in[2];

				  w += 1.f;
			  }
			}

	for (int i = 0; i < 3; ++i)
	{
		pixel_out[i] = col[i] / w;
		//pixel_out[i] = (pixel00[i] + pixel22[i] + pixel02[i] + pixel20[i])*0.125 + (pixel01[i]+pixel10[i]+pixel21[i]+pixel12[i])*0.125 + pixel11[i]*0.0;
	}
	pixel_out[3] = 1; // alpha

}

extern "C"
void texture_blur_progressive(void *source, void *dest, int width, int height, size_t pitch, int step)
{
	dim3 Db = dim3(32, 32);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((width + Db.x - 1) / Db.x, (height + Db.y - 1) / Db.y);

	kernel_blur_progressive << <Dg, Db >> >((unsigned char *)source, (unsigned char *)dest, width, height, pitch, step);
}



__global__ void kernel_transform(unsigned char *source, unsigned char *dest, int width, int height, size_t pitch, float mxx, float mxy, float myx, float myy, float dx, float dy, float zoom)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	float *pixel_out;

	// in the case where, due to quantization into grids, we have
	// more threads than pixels, skip the threads which don't
	// correspond to valid pixels
	if (x >= (width) || y >= (height)) return;

	pixel_out = (float *)(dest + y * pitch) + 4 * x;
	

	float xv = x;
	float yv = y;

	float xw = xv * mxx + yv * mxy + dx;
	float yw = xv * myx + yv * myy + dy;

	float xd = 0.5f + xw - width / 2;
	float yd = 0.5f + yw - height / 2;

	const float h = 1.f;
	const float f = 0.57735f;
	float d = zoom;
	float fh = f * h;
	float dh = d * h;

	float or = sqrt(xd * xd + yd * yd) / width;
	float r = 1.f - or ;

	float nr = (dh*(h - r) + fh * r) / (fh + dh - d * r);

	float onr = 1.f - nr;
	float dst = onr / or;

	xw = xd * dst - 0.5f + width / 2;
	yw = yd * dst - 0.5f + height / 2;

	int lx = floor(xw);
	int ly = floor(yw);

	int hx = lx + 1;
	int hy = ly + 1;


	float *p00 = (float *)(source + ly * pitch) + 4 * lx;
	float *p01 = (float *)(source + ly * pitch) + 4 * hx;
	float *p10 = (float *)(source + hy * pitch) + 4 * lx;
	float *p11 = (float *)(source + hy * pitch) + 4 * hx;

	float wx = xw - lx;
	float wy = yw - ly;

	/*if (lx < 0)
	{
		lx = hx;
	}
	else if (hx >= width)
	{
		hx = lx;
	}


	if (ly < 0)
	{
		ly = hy;
	}
	else if (hy >= height)
	{
		hy = ly;
	}*/

	if ((lx >= 0) && (ly >= 0) && (hx < width) && (hy < height))
	{

		for (int i = 0; i < 3; ++i)
		{
			float c1 = wx * p01[i] + (1.f - wx)*p00[i];
			float c2 = wx * p11[i] + (1.f - wx)*p10[i];

			pixel_out[i] = wy*c2 + (1.f-wy)*c1;
			//pixel_out[i] = 0.5f;
		}
	}
	else
	{
		pixel_out[0] = 0.f;
		pixel_out[1] = 0.f;
		pixel_out[2] = 0.f;
		//for (int i = 0; i < 3; ++i)
		//{
		//	pixel_out[i] = 0.f;
		//}
	}

	
	
	pixel_out[3] = 1; // alpha
}

extern "C"
void transform(void *source, void *dest, int width, int height, size_t pitch, float angle, float dx, float dy, float zoom)
{
	dim3 Db = dim3(32, 32);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((width + Db.x - 1) / Db.x, (height + Db.y - 1) / Db.y);

	float ca = cos(angle);
	float sa = sin(angle);
	float sx = width;
	float sy = height;
	float ndx = (sx*(1.f - ca) - sy*sa) / 2.f + dx;
	float ndy = (sx*sa + sy * (1 - ca)) / 2.f + dy;

	kernel_transform << <Dg, Db >> >((unsigned char *)source, (unsigned char *)dest, width, height, pitch, ca, sa, -sa, ca, ndx, ndy, zoom);
}


__global__ void kernel_compare(unsigned char *source1, unsigned char *source2, unsigned char *dest, int width, int height, size_t pitch)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	float *pixel_out;

	// in the case where, due to quantization into grids, we have
	// more threads than pixels, skip the threads which don't
	// correspond to valid pixels
	if (x >= (width) || y >= (height)) return;

	pixel_out = (float *)(dest + y * pitch) + 4 * x;

	float *p1 = (float *)(source1 + y * pitch) + 4 * x;
	float *p2 = (float *)(source2 + y * pitch) + 4 * x;

	float h1 = (p1[0] + p1[1] + p1[2]) / 3.f;
	float h2 = (p2[0] + p2[1] + p2[2]) / 3.f;
	//float h = (h1*p1[3] + h2 * p2[3]) / min((p1[3] + p2[3]), 1.f);
	float h = (h1 + h2) / 2.f;


	float zr = p1[0] - p2[0];
	float zg = p1[1] - p2[1];
	float zb = p1[2] - p2[2];

	float z = (zr*zr + zg * zg + zb * zb)*2;

	z = sqrt(sqrt(min(z, 1.f)));
	

	pixel_out[0] = h * z;
	pixel_out[2] = 0.f;
	pixel_out[1] = h * (1.f - z);

	pixel_out[3] = 1;
}

extern "C"
void compare(void *source1, void *source2, void *dest, int width, int height, size_t pitch)
{
	dim3 Db = dim3(32, 32);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((width + Db.x - 1) / Db.x, (height + Db.y - 1) / Db.y);

	kernel_compare << <Dg, Db >> >((unsigned char *)source1, (unsigned char *)source2, (unsigned char *)dest, width, height, pitch);
}

__global__ void kernel_multiply(unsigned char *source1, unsigned char *source2, unsigned char *dest, int width, int height, size_t pitch)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	float *pixel_out;

	// in the case where, due to quantization into grids, we have
	// more threads than pixels, skip the threads which don't
	// correspond to valid pixels
	if (x >= (width) || y >= (height)) return;

	pixel_out = (float *)(dest + y * pitch) + 4 * x;

	float *p1 = (float *)(source1 + y * pitch) + 4 * x;
	float *p2 = (float *)(source2 + y * pitch) + 4 * x;

	for(int i=0;i<4;++i)
	pixel_out[i] = p1[i] * p2[i];
}

extern "C"
void multiply(void *source1, void *source2, void *dest, int width, int height, size_t pitch)
{
	dim3 Db = dim3(32, 32);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((width + Db.x - 1) / Db.x, (height + Db.y - 1) / Db.y);

	kernel_multiply << <Dg, Db >> >((unsigned char *)source1, (unsigned char *)source2, (unsigned char *)dest, width, height, pitch);
}



__global__ void kernel_gradient(unsigned char *source, unsigned char *dest_x, unsigned char *dest_y, int width, int height, size_t pitch)
{
	const float coef = 16.f;

	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= (width) || y >= (height)) return;

	int xl = max((int)(x - 1), 0);
	int xh = min((int)(x + 1), width - 1);
	int yl = max((int)(y - 1), 0);
	int yh = min((int)(y + 1), height - 1);

	float* px0 = (float *)(source + y * pitch) + 4 * xl;
	float* px1 = (float *)(source + y * pitch) + 4 * xh;

	float* py0 = (float *)(source + yl * pitch) + 4 * x;
	float* py1 = (float *)(source + yh * pitch) + 4 * x;
	
	float *px = (float *)(dest_x + y * pitch) + 4 * x;
	float *py = (float *)(dest_y + y * pitch) + 4 * x;

	for (int i = 0; i < 3; ++i)
	{
		px[i] = (px1[i] - px0[i])*coef;
		py[i] = (py1[i] - py0[i])*coef;
	}
}


extern "C"
void compute_gradient(void *source, void *dest_x, void *dest_y, int width, int height, size_t pitch)
{
	dim3 Db = dim3(32, 32);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((width + Db.x - 1) / Db.x, (height + Db.y - 1) / Db.y);

	kernel_gradient << <Dg, Db >> >((unsigned char *)source, (unsigned char *)dest_x, (unsigned char *)dest_y, width, height, pitch);
}



__global__ void kernel_force(unsigned char *source, unsigned char *original, unsigned char *grad_x, unsigned char *grad_y,  unsigned char *dest,int width, int height, size_t pitch)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= (width) || y >= (height)) return;

	float* pixel_out = (float *)(dest + y * pitch) + 4 * x;
	
	float* ps = (float *)(source + y * pitch) + 4 * x;
	float* po = (float *)(original + y * pitch) + 4 * x;

	float* gx = (float *)(grad_x + y * pitch) + 4 * x;
	float* gy = (float *)(grad_y + y * pitch) + 4 * x;

	float ox = 0.f;
	float oy = 0.f;

	for (int i = 0; i < 3; ++i)
	{
		ox += (ps[i] - po[i])*gx[i];
		oy += (ps[i] - po[i])*gy[i];
	}

	pixel_out[0] = ox;
	pixel_out[1] = oy;
	pixel_out[2] = 0.0f;
	pixel_out[3] = 1.f;
}


extern "C"
void compute_force(void *source, void *original, void *grad_x, void *grad_y, void *dest, int width, int height, size_t pitch)
{
	dim3 Db = dim3(32, 32);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((width + Db.x - 1) / Db.x, (height + Db.y - 1) / Db.y);

	kernel_force << <Dg, Db >> >((unsigned char *)source, (unsigned char *)original, (unsigned char *)grad_x, (unsigned char *)grad_y, (unsigned char *)dest, width, height, pitch);
}



__global__ void kernel_produce(unsigned char *source, unsigned char *dest, int width, int height, size_t pitch)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	float *pixel_out;
	float *pixel_in;


	// in the case where, due to quantization into grids, we have
	// more threads than pixels, skip the threads which don't
	// correspond to valid pixels
	if (x >= (width) || y >= (height)) return;

	pixel_out = (float *)(dest + y * pitch) + 4 * x;
	pixel_in = (float *)(source + y * pitch) + 4 * x;

	for (int i = 0; i < 3; ++i)
	{
		pixel_out[i] = pixel_in[i] * 0.5f + 0.5f;
	}

	pixel_out[3] = 1; // alpha

}

extern "C"
void texture_produce(void *source, void *dest, int width, int height, size_t pitch)
{
	dim3 Db = dim3(32, 32);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((width + Db.x - 1) / Db.x, (height + Db.y - 1) / Db.y);

	kernel_produce << <Dg, Db >> >((unsigned char *)source, (unsigned char *)dest, width, height, pitch);
}



__global__ void kernel_transform_force(unsigned char *source, unsigned char *dest, int width, int height, size_t pitch, float dx, float dy)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	float *pixel_out;

	if (x >= (width) || y >= (height)) return;

	float vx = x - (width/2 + dx);
	float vy = y - (height / 2 + dy);

	float vortx = vy;
	float vorty = -vx;

	vx /= width;
	vy /= width;

	float r = sqrt(vx*vx + vy*vy);
	float vzx = r * vx;
	float vzy = r * vy;

	pixel_out = (float *)(dest + y * pitch) + 4 * x;
	float *pixel_in = (float *)(source + y * pitch) + 4 * x;

	pixel_out[0] = pixel_in[0];
	pixel_out[1] = pixel_in[1];

	pixel_out[2] = (pixel_in[0]*vortx + pixel_in[1]*vorty)/(width/4);
	pixel_out[3] = (pixel_in[0] * vzx + pixel_in[1] * vzy);
}

extern "C"
void apply_force(void * source, void* dest, int width, int height, size_t pitch, float angle, float dx, float dy)
{
	dim3 Db = dim3(32, 32);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((width + Db.x - 1) / Db.x, (height + Db.y - 1) / Db.y);

	kernel_transform_force << <Dg, Db >> >((unsigned char *)source, (unsigned char *)dest, width, height, pitch, dx, dy);
}

__global__ void kernel_reduce_force(unsigned char *source, unsigned char *dest, int width, int height, int nwidth, int nheight, size_t pitch)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	float *pixel_out;

	if (x >= (nwidth) || y >= (nheight)) return;

	pixel_out = (float *)(dest + y * pitch) + 4 * x;

	int xo = 2 * x;
	int yo = 2 * y;

	float rpix[4];
	
	float* pix = (float *)(source + yo * pitch) + 4 * xo;

	for (int i = 0; i < 4; i++)
		rpix[i] = pix[i];

	if (yo < height - 1)
	{
		pix = (float *)(source + (yo+1) * pitch) + 4 * xo;

		for (int i = 0; i < 4; i++)
			rpix[i] += pix[i];
	}

	if (xo < width - 1)
	{
		pix = (float *)(source + (yo) * pitch) + 4 * (xo + 1);

		for (int i = 0; i < 4; i++)
			rpix[i] += pix[i];

		if (yo < height - 1)
		{
			pix = (float *)(source + (yo + 1) * pitch) + 4 * (xo + 1);

			for (int i = 0; i < 4; i++)
				rpix[i] += pix[i];
		}
	}


	for (int i = 0; i < 4; i++)
	{
		pixel_out[i] = rpix[i] / 4.f;
	}

}

extern "C"
void reduce_force(void * source, void* tmp1, void* tmp2, int width, int height, size_t pitch, int* out_xs, int* out_ys)
{
	dim3 Db = dim3(32, 32);   // block dimensions are fixed to be 256 threads

	int nwidth = width / 2;
	int nheight = height / 2;

	dim3 Dg = dim3((nwidth + Db.x - 1) / Db.x, (nheight + Db.y - 1) / Db.y);

	kernel_reduce_force << <Dg, Db >> > ((unsigned char *)source, (unsigned char *)tmp1, width, height, nwidth, nheight, pitch);


	width = nwidth;
	height = nheight;

	for(int i=0;i<6;i++)
	{
		nwidth = width / 2;
		nheight = height / 2;

		Dg = dim3((nwidth + Db.x - 1) / Db.x, (nheight + Db.y - 1) / Db.y);

		kernel_reduce_force << <Dg, Db >> > ((unsigned char *)tmp1, (unsigned char *)tmp2, width, height, nwidth, nheight, pitch);

		width = nwidth;
		height = nheight;

		source = tmp1;
		tmp1 = tmp2;
		tmp2 = source;
	}

	*out_xs = width;
	*out_ys = height;
}

