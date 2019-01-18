#pragma once

#include "Settings.h"

#include <QVector2D>
#include <cmath>

constexpr double degToRad = 3.14159265358979/180.0;
constexpr double radToDeg = 180.0/3.14159265358979;

class MODELBASIC_EXPORT Physics
{
public:
    //Notice: all angles below are in DEG
    static void collision (QVector2D vA1, QVector2D vB1, QVector2D rAPp, QVector2D rBPp, double angularVelA1
		, double angularVelB1, QVector2D n, double angularMassA, double angularMassB, double massA, double massB
		, QVector2D& vA2, QVector2D& vB2, double& angularVelA2, double& angularVelB2);
    static void fusion (QVector2D vA1, QVector2D vB1, QVector2D rAPp, QVector2D rBPp, double angularVelA1
		, double angularVelB1, QVector2D n, double angularMassA, double angularMassB, double angularMassAB
		, double massA, double massB, QVector2D& v2, double& angularVel2);

    static QVector2D tangentialVelocity (QVector2D positionFromCenter, QVector2D velocityOfCenter, double angularVel);

    static double kineticEnergy (double mass, QVector2D vel, double angularMass, double angularVel);
	static double linearKineticEnergy(double mass, QVector2D vel);
	static double rotationalKineticEnergy(double angularMass, double angularVel);
	static double angularMass(vector<QVector2D> const& relPositionOfMasses);
	static double angularMomentum(QVector2D positionFromCenter, QVector2D velocityOfCenter);
	static double angularVelocity(double angularMassOld, double angularMassNew, double angularVelOld);
    static double angularVelocity (double angularMomentum, double angularMass);

	struct Velocities {
		QVector2D linear;
		double angular;
	};
	static Velocities velocitiesOfCenter(Velocities const& velocities, vector<QVector2D> const& relPositionOfMasses);

    static void applyImpulse (QVector2D const& impulse, QVector2D const& relPos, double mass, QVector2D const& vel, double angularMass
		, double angularVel, QVector2D& newVel, double& newAngularVel);

    //angles are returned in [0,360]
    static QVector2D rotateClockwise (QVector2D v, double angle);
    static QVector2D rotateQuarterCounterClockwise (QVector2D v);
    static double angleOfVector (QVector2D v);   //0 DEG corresponds to (0,-1)
    static QVector2D unitVectorOfAngle (double angle);
    static bool compareEqualAngle (double angle1, double angle2, double precision = Const::AlienPrecision);
    static double subtractAngle (double angleMinuend, double angleSubtrahend);
    static double clockwiseAngleFromFirstToSecondVector (QVector2D v1, QVector2D v2);
};
