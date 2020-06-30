#include "SpaceProperties.h"

SpaceProperties::SpaceProperties(QObject * parent)
	: QObject(parent)
{
}

void SpaceProperties::init(IntVector2D size)
{
	_size = size;
}

SpaceProperties * SpaceProperties::clone(QObject * parent) const
{
	auto metric = new SpaceProperties(parent);
	metric->_size = _size;
	return metric;
}

IntVector2D SpaceProperties::getSize() const
{
	return _size;
}

void SpaceProperties::correctPosition(QVector2D & pos) const
{
	IntVector2D intPart{ qFloor(pos.x()), qFloor(pos.y()) };
	qreal fracPartX = pos.x() - intPart.x;
	qreal fracPartY = pos.y() - intPart.y;
	correctPositionInline(intPart);
	pos.setX(static_cast<qreal>(intPart.x) + fracPartX);
	pos.setY(static_cast<qreal>(intPart.y) + fracPartY);
}

void SpaceProperties::correctPosition(IntVector2D & pos) const
{
	correctPositionInline(pos);
}

IntVector2D SpaceProperties::convertToIntVector(QVector2D const & pos) const
{
	IntVector2D intPos;
	intPos.x = static_cast<int>(pos.x());
	if (intPos.x < 0) {
		--intPos.x;
	}
	intPos.y = static_cast<int>(pos.y());
	if (intPos.y < 0) {
		--intPos.y;
	}
	return intPos;
}

IntVector2D SpaceProperties::correctPositionAndConvertToIntVector(QVector2D const& pos) const
{
	IntVector2D intPos = convertToIntVector(pos);
	correctPositionInline(intPos);
	return intPos;
}

IntVector2D SpaceProperties::shiftPosition(IntVector2D const & pos, IntVector2D const && shift) const
{
	IntVector2D temp{ pos.x + shift.x, pos.y + shift.y };
	correctPositionInline(temp);
	return temp;
}

void SpaceProperties::truncateRect(IntRect & rect) const
{
	truncatePosition(rect.p1);
	truncatePosition(rect.p2);
}

void SpaceProperties::correctDisplacement(QVector2D & displacement) const
{
	IntVector2D intDisplacement{ qFloor(displacement.x()), qFloor(displacement.y()) };
	qreal rx = displacement.x() - static_cast<qreal>(intDisplacement.x);
	qreal ry = displacement.y() - static_cast<qreal>(intDisplacement.y);
	intDisplacement.x += _size.x / 2;
	intDisplacement.y += _size.y / 2;
	correctPositionInline(intDisplacement);
	intDisplacement.x -= _size.x / 2;
	intDisplacement.y -= _size.y / 2;
	displacement.setX(static_cast<qreal>(intDisplacement.x) + rx);
	displacement.setY(static_cast<qreal>(intDisplacement.y) + ry);
}

QVector2D SpaceProperties::displacement(QVector2D fromPoint, QVector2D toPoint) const
{
	QVector2D d = toPoint - fromPoint;
	correctDisplacement(d);
	return d;
}

qreal SpaceProperties::distance(QVector2D fromPoint, QVector2D toPoint) const
{
	return displacement(fromPoint, toPoint).length();
}

QVector2D SpaceProperties::correctionIncrement(QVector2D pos1, QVector2D pos2) const
{
	QVector2D correction;
	if ((pos2.x() - pos1.x()) > (_size.x / 2.0))
		correction.setX(-_size.x);
	if ((pos1.x() - pos2.x()) > (_size.x / 2.0))
		correction.setX(_size.x);
	if ((pos2.y() - pos1.y()) > (_size.y / 2.0))
		correction.setY(-_size.y);
	if ((pos1.y() - pos2.y()) > (_size.y / 2.0))
		correction.setY(_size.y);
	return correction;
}

void SpaceProperties::truncatePosition(IntVector2D & pos) const
{
	pos.x = pos.x < 0 ? 0 : pos.x;
	pos.y = pos.y < 0 ? 0 : pos.y;
	pos.x = pos.x >= _size.x ? _size.x - 1 : pos.x;
	pos.y = pos.y >= _size.y ? _size.y - 1 : pos.y;
}

