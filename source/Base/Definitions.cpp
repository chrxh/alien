#include "Definitions.h"

IntVector2D::IntVector2D()
	: x(0), y(0)
{
}

IntVector2D::IntVector2D(std::initializer_list<int> l)
{
	auto it = l.begin();
	x = *it++;
	y = *it;
}

IntVector2D::IntVector2D(QVector2D const& vec)
	: x(static_cast<int>(vec.x())), y(static_cast<int>(vec.y()))
{
}

QVector2D IntVector2D::toQVector2D()
{
	return QVector2D(x, y); 
}

IntVector2D & IntVector2D::restrictToRect(IntRect const & rect)
{
	if (x < rect.p1.x) {
		x = rect.p1.x;
	}
	if (y < rect.p1.y) {
		y = rect.p1.y;
	}
	if (x > rect.p2.x) {
		x = rect.p2.x;
	}
	if (y > rect.p2.y) {
		y = rect.p2.y;
	}
	return *this;
}

bool IntVector2D::operator==(IntVector2D const & vec)
{
	return x == vec.x && y == vec.y;
}

std::ostream& operator << (std::ostream& os, const IntVector2D& vec)
{
	os << "(" << vec.x << ", " << vec.y << ")";
	return os;
}


IntRect::IntRect(std::initializer_list<IntVector2D> l)
{
	auto it = l.begin();
	p1 = *it++;
	p2 = *it;
}

IntRect::IntRect(QRectF const &rect)
	: p1({ static_cast<int>(rect.left()), static_cast<int>(rect.top()) })
	, p2({ static_cast<int>(rect.right()), static_cast<int>(rect.bottom()) })
{
}

