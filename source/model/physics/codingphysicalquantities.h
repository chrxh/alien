#ifndef CodingPhysicalQuantities_H
#define CodingPhysicalQuantities_H

#include <QtGlobal>

class CodingPhysicalQuantities
{
public:
	//Notice: all angles below are in DEG
	static qreal convertDataToAngle(quint8 b);
	static quint8 convertAngleToData(qreal a);
	static qreal convertDataToShiftLen(quint8 b);
	static quint8 convertShiftLenToData(qreal len);
	static quint8 convertURealToData(qreal r);
	static qreal convertDataToUReal(quint8 d);
	static quint8 convertIntToData(int i);
};

#endif // CodingPhysicalQuantities_H
