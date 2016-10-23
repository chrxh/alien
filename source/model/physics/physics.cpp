#include "physics.h"

#include <qmath.h>
#include <QMatrix4x4>

//calculate collision of two moving and rotating rigid bodies
void Physics::collision (QVector3D vA1, QVector3D vB1, QVector3D rAPp, QVector3D rBPp,
                         /*QVector3D posA, QVector3D posB, QVector3D posP, */qreal angularVelA1,
                         qreal angularVelB1, QVector3D n, qreal angularMassA, qreal angularMassB,
                         qreal massA, qreal massB,
                         QVector3D& vA2, QVector3D& vB2, qreal& angularVelA2, qreal& angularVelB2)
{
    angularVelA1 *= degToRad;
    angularVelB1 *= degToRad;

    if( (qAbs(angularMassA) <= ALIEN_PRECISION) )
        angularVelA1 = 0.0;
    if( (qAbs(angularMassB) <= ALIEN_PRECISION) )
        angularVelB1 = 0.0;

    QVector3D vAB = vA1-rAPp*angularVelA1-(vB1-rBPp*angularVelB1);
    if( QVector3D::dotProduct(vAB,n) > 0.0 ) {
        vA2 = vA1;
        angularVelA2 = angularVelA1;
        vB2 = vB1;
        angularVelB2 = angularVelB1;
    }
    else {
        if( (qAbs(angularMassA) > ALIEN_PRECISION) && (qAbs(angularMassB) > ALIEN_PRECISION) ) {
            qreal j=-2.0*QVector3D::dotProduct(vAB,n)/(n.lengthSquared()*(1.0/massA+1.0/massB)
                    +QVector3D::dotProduct(rAPp,n)*QVector3D::dotProduct(rAPp,n)/angularMassA
                    +QVector3D::dotProduct(rBPp,n)*QVector3D::dotProduct(rBPp,n)/angularMassB);
            vA2 = vA1 + j/massA*n;
            angularVelA2 = angularVelA1-QVector3D::dotProduct(rAPp,n)*j/angularMassA;
            vB2 = vB1 - j/massB*n;
            angularVelB2 = angularVelB1+QVector3D::dotProduct(rBPp,n)*j/angularMassB;
        }
        if( (qAbs(angularMassA) <= ALIEN_PRECISION) && (qAbs(angularMassB) > ALIEN_PRECISION) ) {
            qreal j=-2.0*QVector3D::dotProduct(vAB,n)/(n.lengthSquared()*(1.0/massA+1.0/massB)
                    +QVector3D::dotProduct(rBPp,n)*QVector3D::dotProduct(rBPp,n)/angularMassB);
            vA2 = vA1 + j/massA*n;
            angularVelA2 = angularVelA1;
            vB2 = vB1 - j/massB*n;
            angularVelB2 = angularVelB1+QVector3D::dotProduct(rBPp,n)*j/angularMassB;
        }
        if( (qAbs(angularMassA) > ALIEN_PRECISION) && (qAbs(angularMassB) <= ALIEN_PRECISION) ) {
            qreal j=-2.0*QVector3D::dotProduct(vAB,n)/(n.lengthSquared()*(1.0/massA+1.0/massB)
                    +QVector3D::dotProduct(rAPp,n)*QVector3D::dotProduct(rAPp,n)/angularMassA);
            vA2 = vA1 + j/massA*n;
            angularVelA2 = angularVelA1-QVector3D::dotProduct(rAPp,n)*j/angularMassA;
            vB2 = vB1 - j/massB*n;
            angularVelB2 = angularVelB1;
        }
        if( (qAbs(angularMassA) <= ALIEN_PRECISION) && (qAbs(angularMassB) <= ALIEN_PRECISION) ) {
            qreal j=-2.0*QVector3D::dotProduct(vAB,n)/(n.lengthSquared()*(1.0/massA+1.0/massB));
            vA2 = vA1 + j/massA*n;
            angularVelA2 = angularVelA1;
            vB2 = vB1 - j/massB*n;
            angularVelB2 = angularVelB1;
        }
    }
    angularVelA2 *= radToDeg;
    angularVelB2 *= radToDeg;
}

void Physics::fusion (QVector3D vA1, QVector3D vB1, QVector3D rAPp, QVector3D rBPp,
                      /*QVector3D posA, QVector3D posB, QVector3D posP, */qreal angularVelA1,
                      qreal angularVelB1, QVector3D n, qreal angularMassA, qreal angularMassB, qreal angularMassAB,
                      qreal massA, qreal massB,
                      QVector3D& v2, qreal& angularVel2)
{
    //calculation of rAPp
    /*QVector3D rAPp = posP-posA;
    qreal temp = rAPp.x();
    rAPp.setX(rAPp.y());
    rAPp.setY(-temp);
    QVector3D rBPp = posP-posB;
    temp = rBPp.x();
    rBPp.setX(rBPp.y());
    rBPp.setY(-temp);*/

    QVector3D vA2;
    QVector3D vB2;
    qreal angularVelA2(0.0);
    qreal angularVelB2(0.0);

    Physics::collision(vA1, vB1, rAPp, rBPp, angularVelA1,
                       angularVelB1, n, angularMassA, angularMassB,
                       massA, massB,
                       vA2, vB2, angularVelA2, angularVelB2);
    v2 = (vA2*massA+vB2*massB)/(massA+massB);
    angularVel2 = (angularVelA2*angularMassA+angularVelB2*angularMassB)/angularMassAB;
}


//calculate velocity and angular velocity for new center via energies
void Physics::changeCenterOfMass (qreal mass, QVector3D vel, qreal angularVel, qreal oldAngularMass,
                              qreal newAngularMass, QVector3D centerDiff,
                              QVector3D& newVel, qreal& newAngularVel)
{
    newAngularVel = angularVel;
    newVel = tangentialVelocity(centerDiff, vel, angularVel);
    angularVel *= degToRad;
    qreal kinEnergy = vel.lengthSquared()*mass+(oldAngularMass-newAngularMass)*angularVel*angularVel;
    if( kinEnergy >= ALIEN_PRECISION ) {
        qreal newVelAbs = qSqrt(kinEnergy / mass);
        newVel = newVel.normalized()*newVelAbs;
    }
}

//remember that the coordinate system in a computer system is mirrored at the x-axis...
QVector3D Physics::tangentialVelocity (QVector3D r, QVector3D vel, qreal angularVel)
{
    return vel - angularVel*rotateQuarterCounterClockwise(r)*degToRad;
}


qreal Physics::kineticEnergy (qreal mass, QVector3D vel, qreal angularMass, qreal angularVel)
{
    angularVel *= degToRad;
    return 0.5*angularMass*angularVel*angularVel+0.5*mass*vel.lengthSquared();
}

qreal Physics::newAngularVelocity (qreal angularMassOld, qreal angularMassNew, qreal angularVelOld)
{
    angularVelOld = angularVelOld*degToRad;
    qreal rotEnergyDouble = angularMassOld*angularVelOld*angularVelOld;
    qreal angularVelNew = qSqrt(rotEnergyDouble/angularMassNew);
    return angularVelNew*radToDeg;
}

qreal Physics::newAngularVelocity2 (qreal Ekin, qreal Etrans, qreal angularMass, qreal angularVelOld)
{
    qreal Erot = Ekin - Etrans;
    if( Erot > 0.0 ) {
        qreal angularVelNew = qSqrt(2.0*Erot/angularMass)*radToDeg;
        if( angularVelOld > 0.0 )
            return qAbs(angularVelNew);
        else
            return -qAbs(angularVelNew);
    }
    else {
        return 0.0;
    }
}

qreal Physics::angularMomentum (QVector3D r, QVector3D v)
{
    return r.x()*v.y()-r.y()*v.x();     //we only calc the 3rd component of the 3D cross product
}

qreal Physics::angularVelocity (qreal angularMomentum, qreal angularMass)
{
    if( qAbs(angularMass) < ALIEN_PRECISION )
        return 0;
    else
        return angularMomentum/angularMass*radToDeg;
}

void Physics::applyImpulse (QVector3D impulse, QVector3D rAPp, qreal mass, QVector3D vel, qreal angularMass, qreal angularVel, QVector3D& newVel, qreal& newAngularVel)
{
    newVel = vel + impulse/mass;
    newAngularVel = angularVel - QVector3D::dotProduct(rAPp, impulse)/angularMass*radToDeg;
}

QVector3D Physics::rotateClockwise (QVector3D v, qreal angle)
{
    QMatrix4x4 transform;
    transform.rotate(angle, 0.0, 0.0, 1.0);
    return transform.map(v);
}

QVector3D Physics::rotateQuarterCounterClockwise (QVector3D v)
{

    //90 degree counterclockwise rotation of vector r
    qreal temp(v.x());
    v.setX(v.y());
    v.setY(-temp);
    return v;
}

qreal Physics::angleOfVector (QVector3D v)
{
    qreal angle(0.0);
    qreal angleSin(qAsin(-v.y()/v.length())*radToDeg);
    if( v.x() >= 0.0 )
        angle = 90.0-angleSin;
    else
        angle = angleSin+270.0;
    return angle;
}

QVector3D Physics::unitVectorOfAngle (qreal angle)
{
    return QVector3D(qSin(angle*degToRad),-qCos(angle*degToRad), 0.0);
}

bool Physics::compareEqualAngle (qreal angle1, qreal angle2, qreal precision)
{
    if( qAbs(angle1-angle2) < precision )
        return true;
    if( qAbs(angle1-angle2-360.0) < precision )
        return true;
    if( qAbs(angle1-angle2+360.0) < precision )
        return true;
    return false;
}

qreal Physics::subtractAngle (qreal angleMinuend, qreal angleSubtrahend)
{
    qreal angleDiff = angleMinuend- angleSubtrahend;
    if( angleDiff > 360.0 )
        angleDiff -= 360.0;
    if( angleDiff < 0.0 )
        angleDiff += 360.0;
    return angleDiff;
}

qreal Physics::clockwiseAngleFromFirstToSecondVector (QVector3D v1, QVector3D v2)
{
    qreal a1 = Physics::angleOfVector(v1);
    qreal a2 = Physics::angleOfVector(v2);
    return subtractAngle(a2, a1);
}
