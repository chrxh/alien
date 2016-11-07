#include "aliencellfunction.h"

#include "model/entities/aliengrid.h"
#include "model/physics/physics.h"

#include <QString>
#include <QtCore/qmath.h>

qreal AlienCellFunction::calcAngle (AlienCell* origin, AlienCell* ref1, AlienCell* ref2) const
{
    QVector3D v1 = _grid->displacement(origin, ref1);
    QVector3D v2 = _grid->displacement(origin, ref2);
    return Physics::clockwiseAngleFromFirstToSecondVector(v1, v2);
}

qreal AlienCellFunction::convertDataToAngle (quint8 b)
{
    //0 to 127 => 0 to 179 degree
    //128 to 255 => -179 to 0 degree
    if( b < 128 )
        return (0.5+static_cast<qreal>(b))*(180.0/128.0);        //add 0.5 to prevent discretization errors
    else
        return (-256.0-0.5+static_cast<qreal>(b))*(180.0/128.0);
}

quint8 AlienCellFunction::convertAngleToData (qreal a)
{
    //0 to 180 degree => 0 to 128
    //-180 to 0 degree => 128 to 256 (= 0)
    if( a > 180.0 )
        a = a - 360.0;
    if( a <= -180.0 )
        a = a + 360.0;
    int intA = static_cast<int>(a*128.0/180.0);
    return static_cast<quint8>(intA);
}

qreal AlienCellFunction::convertDataToShiftLen (quint8 b)
{
    return (0.5+(qreal)b)/100.0;                    //add 0.5 to prevent discretization errors
}

quint8 AlienCellFunction::convertShiftLenToData (qreal len)
{
    if( static_cast< quint32 >(len*100.0) >= 256 )
        return 255;
    return static_cast< quint8 >(len*100.0);
}

quint8 AlienCellFunction::convertURealToData (qreal r)
{
    if( r < 0.0 )
        return 0;
    if( r > 127.0)
        return 127;
    return qFloor(r);
}

qreal AlienCellFunction::convertDataToUReal (quint8 d)
{
    return static_cast< qreal >(d);
}

quint8 AlienCellFunction::convertIntToData (int i)
{
    if( i > 127 )
        return i;
    if( i < -128)
        return -128;
    return i;
}

