#ifndef PHYSICS_H
#define PHYSICS_H

#include <QVector3D>
#include <cmath>

constexpr qreal degToRad = 3.14159265358979/180.0;
constexpr qreal radToDeg = 180.0/3.14159265358979;


class Physics
{
public:
    static void collision (QVector3D vA1, QVector3D vB1, QVector3D rAPp, QVector3D rBPp, /*QVector3D posA, QVector3D posB, QVector3D posP, */qreal angularVelA1,
                           qreal angularVelB1, QVector3D n, qreal angularMassA, qreal angularMassB,
                           qreal massA, qreal massB,
                           QVector3D& vA2, QVector3D& vB2, qreal& angularVelA2, qreal& angularVelB2);

    static void fusion (QVector3D vA1, QVector3D vB1, QVector3D rAPp, QVector3D rBPp, /*QVector3D posA, QVector3D posB, QVector3D posP, */qreal angularVelA1,
                           qreal angularVelB1, QVector3D n, qreal angularMassA, qreal angularMassB, qreal angularMassAB,
                           qreal massA, qreal massB,
                           QVector3D& v2, qreal& angularVel2);

    static void changeCenterOfMass (qreal mass, QVector3D vel, qreal angularVel, qreal oldAngularMass,
                                         qreal newAngularMass, QVector3D centerDiff,
                                         QVector3D& newVel, qreal& newAngularVel);

    static QVector3D calcTangentialVelocity (QVector3D r, QVector3D vel, qreal angularVel);

    static qreal calcKineticEnergy (qreal mass, QVector3D vel, qreal angularMass, qreal angularVel);

    static qreal calcNewAngularVelocity (qreal angularMassOld, qreal angularMassNew, qreal angularVelOld);  //returns new angular velocity

    static qreal calcNewAngularVelocity2 (qreal Ekin, qreal Etrans, qreal angularMass, qreal angularVelOld);   //returns new angular velocity

    static qreal calcAngularMomentum (QVector3D r, QVector3D v);

    static qreal calcAngularVelocity (qreal angularMomentum, qreal angularMass);

    static void applyImpulse (QVector3D impulse, QVector3D rAPp, qreal mass, QVector3D vel, qreal angularMass, qreal angularVel, QVector3D& newVel, qreal& newAngularVel);

    static QVector3D rotateQuarterCounterClockwise (QVector3D v);

    static qreal calcAngle (QVector3D v);
    static QVector3D calcVector (qreal angle);
};

#endif // PHYSICS_H
