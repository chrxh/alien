#include "PhysicalQuantityConverter.h"

#include <QtCore/qmath.h>

qreal PhysicalQuantityConverter::convertDataToAngle(quint8 b)
{
	//0 to 127 => 0 to 179 degree
	//128 to 255 => -179 to 0 degree
	if (b < 128)
		return (0.5 + static_cast<qreal>(b))*(180.0 / 128.0);        //add 0.5 to prevent discretization errors
	else
		return (-256.0 - 0.5 + static_cast<qreal>(b))*(180.0 / 128.0);
}

quint8 PhysicalQuantityConverter::convertAngleToData(qreal a)
{
	//0 to 180 degree => 0 to 128
	//-180 to 0 degree => 128 to 256 (= 0)
	if (a > 180.0)
		a = a - 360.0;
	if (a <= -180.0)
		a = a + 360.0;
	int intA = static_cast<int>(a*128.0 / 180.0);
	return static_cast<quint8>(intA);
}

qreal PhysicalQuantityConverter::convertDataToShiftLen(quint8 b)
{
	return (0.5 + (qreal)b) / 100.0;                    //add 0.5 to prevent discretization errors
}

quint8 PhysicalQuantityConverter::convertShiftLenToData(qreal len)
{
	if (static_cast<quint32>(len*100.0) >= 256)
		return 255;
	return static_cast<quint8>(len*100.0);
}

quint8 PhysicalQuantityConverter::convertURealToData(qreal r)
{
	if (r < 0.0)
		return 0;
	if (r > 127.0)
		return 127;
	return qFloor(r);
}

qreal PhysicalQuantityConverter::convertDataToUReal(quint8 d)
{
	return static_cast<qreal>(d);
}

quint8 PhysicalQuantityConverter::convertIntToData(int i)
{
	if (i > 127)
		return i;
	if (i < -128)
		return -128;
	return i;
}

