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

bool IntVector2D::operator==(IntVector2D const & vec)
{
	return x == vec.x && y == vec.y;
}

std::ostream& operator << (std::ostream& os, const IntVector2D& vec)
{
	os << "(" << vec.x << ", " << vec.y << ")";
	return os;
}

bool IntRect::isContained(IntVector2D p)
{
	return p1.x <= p.x && p1.y <= p.y && p.x <= p2.x && p.y <= p2.y;
}
