#pragma once
#include <functional>
#include <initializer_list>

template<typename T, int N> class valtuple
{
	T values[N];
public:
	valtuple() = default;
	valtuple(std::initializer_list<float> il) 
	{	std::copy(il.begin(), il.end(), std::begin(values)); }

	template<typename F> void fill(F f)
	{ for (int i = 0; i < N; ++i) values[i] = f(i); }

	T& operator[](int i) { return values[i]; }
	const T& operator[](int i) const { return values[i]; }

	template<typename F> valtuple transform(F f)
	{
		valtuple tmp; tmp.fill([&](int i) { return f(values[i]); }); return tmp;
	}

	template<typename F> valtuple merge(const valtuple& v, F f)
	{
		valtuple tmp; tmp.fill([&](int i) { return f(values[i], v.values[i]); }); return tmp;
	}

	valtuple operator+(const valtuple& v)
	{ return merge(v, std::plus<>{}); }

	valtuple operator-(const valtuple& v)
	{ return merge(v, std::minus<>{}); }

	valtuple operator*(const valtuple& v)
	{ return merge(v, std::multiply<>{});	}

	valtuple operator/(const valtuple& v)
	{  return merge(v, std::divide<>{});	}

	valtuple operator-()
	{ return transform(std::negate<>{}); }

	valtuple operator*(T f)
	{ return transform([f](T x) {return x * f; }); }

	valtuple operator/(T f)
	{ return transform([f](T x) {return x / f; }); }
};