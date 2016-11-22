#ifndef PHYSICS_H
#define PHYSICS_H

#include "model/simulationsettings.h"

#include <QVector3D>
#include <cmath>

constexpr qreal degToRad = 3.14159265358979/180.0;
constexpr qreal radToDeg = 180.0/3.14159265358979;

class Physics
{
public:
    //Notice: all angles below are in DEG

    static void collision (QVector3D vA1, QVector3D vB1, QVector3D rAPp, QVector3D rBPp, qreal angularVelA1,
                           qreal angularVelB1, QVector3D n, qreal angularMassA, qreal angularMassB,
                           qreal massA, qreal massB,
                           QVector3D& vA2, QVector3D& vB2, qreal& angularVelA2, qreal& angularVelB2, int temp = 0); //TEMP
    static void fusion (QVector3D vA1, QVector3D vB1, QVector3D rAPp, QVector3D rBPp, qreal angularVelA1,
                           qreal angularVelB1, QVector3D n, qreal angularMassA, qreal angularMassB, qreal angularMassAB,
                           qreal massA, qreal massB,
                           QVector3D& v2, qreal& angularVel2);

    static void changeCenterOfMass (qreal mass, QVector3D vel, qreal angularVel, qreal oldAngularMass,
                                         qreal newAngularMass, QVector3D centerDiff,
                                         QVector3D& newVel, qreal& newAngularVel);

    static QVector3D tangentialVelocity (QVector3D r, QVector3D vel, qreal angularVel);

    static qreal kineticEnergy (qreal mass, QVector3D vel, qreal angularMass, qreal angularVel);
    static qreal newAngularVelocity (qreal angularMassOld, qreal angularMassNew, qreal angularVelOld);
    static qreal newAngularVelocity2 (qreal Ekin, qreal Etrans, qreal angularMass, qreal angularVelOld);
    static qreal angularMomentum (QVector3D r, QVector3D v);
    static qreal angularVelocity (qreal angularMomentum, qreal angularMass);

    static void applyImpulse (QVector3D impulse, QVector3D rAPp, qreal mass, QVector3D vel, qreal angularMass, qreal angularVel, QVector3D& newVel, qreal& newAngularVel);

    //angles are returned in [0,360]
    static QVector3D rotateClockwise (QVector3D v, qreal angle);
    static QVector3D rotateQuarterCounterClockwise (QVector3D v);
    static qreal angleOfVector (QVector3D v);   //0 DEG corresponds to (0,-1)
    static QVector3D unitVectorOfAngle (qreal angle);
    static bool compareEqualAngle (qreal angle1, qreal angle2, qreal precision = ALIEN_PRECISION);
    static qreal subtractAngle (qreal angleMinuend, qreal angleSubtrahend);
    static qreal clockwiseAngleFromFirstToSecondVector (QVector3D v1, QVector3D v2);
};

#endif // PHYSICS_H
