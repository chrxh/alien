#pragma once

#include "Settings.h"

#include <QVector2D>
#include <cmath>

constexpr qreal degToRad = 3.14159265358979/180.0;
constexpr qreal radToDeg = 180.0/3.14159265358979;

class MODELBASIC_EXPORT CudaPhysics
{
public:
    //Notice: all angles below are in DEG

    static void collision (QVector2D vA1, QVector2D vB1, QVector2D rAPp, QVector2D rBPp, qreal angularVelA1
		, qreal angularVelB1, QVector2D n, qreal angularMassA, qreal angularMassB, qreal massA, qreal massB
		, QVector2D& vA2, QVector2D& vB2, qreal& angularVelA2, qreal& angularVelB2);
    static void fusion (QVector2D vA1, QVector2D vB1, QVector2D rAPp, QVector2D rBPp, qreal angularVelA1
		, qreal angularVelB1, QVector2D n, qreal angularMassA, qreal angularMassB, qreal angularMassAB
		, qreal massA, qreal massB, QVector2D& v2, qreal& angularVel2);

    static void changeCenterOfMass (qreal mass, QVector2D vel, qreal angularVel, qreal oldAngularMass
		, qreal newAngularMass, QVector2D centerDiff, QVector2D& newVel, qreal& newAngularVel);

    static QVector2D tangentialVelocity (QVector2D r, QVector2D vel, qreal angularVel);

    static double kineticEnergy (qreal mass, QVector2D vel, qreal angularMass, qreal angularVel);
	static double linearKineticEnergy(qreal mass, QVector2D vel);
	static double rotationalKineticEnergy(qreal angularMass, qreal angularVel);
	static qreal newAngularVelocity(qreal angularMassOld, qreal angularMassNew, qreal angularVelOld);
    static qreal newAngularVelocity2 (qreal Ekin, qreal Etrans, qreal angularMass, qreal angularVelOld);
    static qreal angularMomentum (QVector2D r, QVector2D v);
    static qreal angularVelocity (qreal angularMomentum, qreal angularMass);

    static void applyImpulse (QVector2D const& impulse, QVector2D const& relPos, qreal mass, QVector2D const& vel, qreal angularMass
		, qreal angularVel, QVector2D& newVel, qreal& newAngularVel);

    //angles are returned in [0,360]
    static QVector2D rotateClockwise (QVector2D v, qreal angle);
    static QVector2D rotateQuarterCounterClockwise (QVector2D v);
    static qreal angleOfVector (QVector2D v);   //0 DEG corresponds to (0,-1)
    static QVector2D unitVectorOfAngle (qreal angle);
    static bool compareEqualAngle (qreal angle1, qreal angle2, qreal precision = Const::AlienPrecision);
    static qreal subtractAngle (qreal angleMinuend, qreal angleSubtrahend);
    static qreal clockwiseAngleFromFirstToSecondVector (QVector2D v1, QVector2D v2);
};
