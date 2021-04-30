#pragma once
#define _USE_MATH_DEFINES
#include<math.h>

template<class R>
void copyv3(const R* s, R* d)
{
	for (int i = 0; i < 3; i++)
	{
		d[i] = s[i];
	}
}

template<class R>
void addv3(const R* a, const R* b, R* c)
{
	for (int i = 0; i < 3; i++)
	{
		c[i] = a[i] + b[i];
	}
}

template<class R>
void minusv3(const R* a, const R* b, R* c)
{
	for (int i = 0; i < 3; i++)
	{
		c[i] = a[i] - b[i];
	}
}

template<class R>
R dotv3(const R* a, const R* b)
{
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

template<class R>
void crossv3(const R* a, const R* b, R* c)
{
	c[0] = a[1] * b[2] - a[2] * b[1];
	c[1] = a[2] * b[0] - a[0] * b[2];
	c[2] = a[0] * b[1] - a[1] * b[0];
}

template<class R>
R mixed_product(const R* a, const R* b, const R* c)
{
	R temp[3];
	crossv3(a, b, temp);
	return dotv3(temp, c);
}

template<class R>
R normv3(const R* v)
{
	return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

template<class R>
void normalizedv3(R* v)
{
	R n = normv3(v);
	for (int i = 0; i < 3; i++)
	{
		v[i] /= n;
	}
}

template<class R>
void scalv3(const R s, R* v)
{
	for (int i = 0; i < 3; i++)
	{
		v[i] *= s;
	}
}

template<class R>
R calc_omega(const R* e1, const R* e2, const R* e3)
{
	R a, b, c;
	a = normv3(e1);
	b = normv3(e2);
	c = normv3(e3);

	R d1, d2, d3;
	d1 = dotv3(e1, e2);
	d2 = dotv3(e3, e1);
	d3 = dotv3(e2, e3);
	return 2 * atan2(mixed_product(e2, e3, e1), a * b * c + d1 * c + d2 * b + d3 * a);
}
