#include "Physics.h"

#include <qmath.h>
#include <QMatrix4x4>

//calculate collision of two moving and rotating rigid bodies
void Physics::collision (QVector2D vA1, QVector2D vB1, QVector2D rAPp, QVector2D rBPp,
                         /*QVector2D posA, QVector2D posB, QVector2D posP, */double angularVelA1,
                         double angularVelB1, QVector2D n, double angularMassA, double angularMassB,
                         double massA, double massB,
                         QVector2D& vA2, QVector2D& vB2, double& angularVelA2, double& angularVelB2)
{
    angularVelA1 *= degToRad;
    angularVelB1 *= degToRad;

    if( (qAbs(angularMassA) <= FLOATINGPOINT_HIGH_PRECISION) )
        angularVelA1 = 0.0;
    if( (qAbs(angularMassB) <= FLOATINGPOINT_HIGH_PRECISION) )
        angularVelB1 = 0.0;

    QVector2D vAB = vA1-rAPp*angularVelA1-(vB1-rBPp*angularVelB1);
    if( QVector2D::dotProduct(vAB,n) > 0.0 ) {
        vA2 = vA1;
        angularVelA2 = angularVelA1;
        vB2 = vB1;
        angularVelB2 = angularVelB1;
    }
    else {
        if( (qAbs(angularMassA) > FLOATINGPOINT_HIGH_PRECISION) && (qAbs(angularMassB) > FLOATINGPOINT_HIGH_PRECISION) ) {
            double j=-2.0*QVector2D::dotProduct(vAB,n)/(n.lengthSquared()*(1.0/massA+1.0/massB)
                    +QVector2D::dotProduct(rAPp,n)*QVector2D::dotProduct(rAPp,n)/angularMassA
                    +QVector2D::dotProduct(rBPp,n)*QVector2D::dotProduct(rBPp,n)/angularMassB);
            vA2 = vA1 + j/massA*n;
            angularVelA2 = angularVelA1-QVector2D::dotProduct(rAPp,n)*j/angularMassA;
            vB2 = vB1 - j/massB*n;
            angularVelB2 = angularVelB1+QVector2D::dotProduct(rBPp,n)*j/angularMassB;
        }
        if( (qAbs(angularMassA) <= FLOATINGPOINT_HIGH_PRECISION) && (qAbs(angularMassB) > FLOATINGPOINT_HIGH_PRECISION) ) {
            double j=-2.0*QVector2D::dotProduct(vAB,n)/(n.lengthSquared()*(1.0/massA+1.0/massB)
                    +QVector2D::dotProduct(rBPp,n)*QVector2D::dotProduct(rBPp,n)/angularMassB);
            vA2 = vA1 + j/massA*n;
            angularVelA2 = angularVelA1;
            vB2 = vB1 - j/massB*n;
            angularVelB2 = angularVelB1+QVector2D::dotProduct(rBPp,n)*j/angularMassB;
        }
        if( (qAbs(angularMassA) > FLOATINGPOINT_HIGH_PRECISION) && (qAbs(angularMassB) <= FLOATINGPOINT_HIGH_PRECISION) ) {
            double j=-2.0*QVector2D::dotProduct(vAB,n)/(n.lengthSquared()*(1.0/massA+1.0/massB)
                    +QVector2D::dotProduct(rAPp,n)*QVector2D::dotProduct(rAPp,n)/angularMassA);
            vA2 = vA1 + j/massA*n;
            angularVelA2 = angularVelA1-QVector2D::dotProduct(rAPp,n)*j/angularMassA;
            vB2 = vB1 - j/massB*n;
            angularVelB2 = angularVelB1;
        }
        if( (qAbs(angularMassA) <= FLOATINGPOINT_HIGH_PRECISION) && (qAbs(angularMassB) <= FLOATINGPOINT_HIGH_PRECISION) ) {
            double j=-2.0*QVector2D::dotProduct(vAB,n)/(n.lengthSquared()*(1.0/massA+1.0/massB));
            vA2 = vA1 + j/massA*n;
            angularVelA2 = angularVelA1;
            vB2 = vB1 - j/massB*n;
            angularVelB2 = angularVelB1;
        }
    }
    angularVelA2 *= radToDeg;
    angularVelB2 *= radToDeg;
}

void Physics::fusion (QVector2D vA1, QVector2D vB1, QVector2D rAPp, QVector2D rBPp, double angularVelA1
	, double angularVelB1, QVector2D n, double angularMassA, double angularMassB, double angularMassAB
	, double massA, double massB, QVector2D& v2, double& angularVel2)
{
    //calculation of rAPp
    /*QVector2D rAPp = posP-posA;
    double temp = rAPp.x();
    rAPp.setX(rAPp.y());
    rAPp.setY(-temp);
    QVector2D rBPp = posP-posB;
    temp = rBPp.x();
    rBPp.setX(rBPp.y());
    rBPp.setY(-temp);*/

    QVector2D vA2;
    QVector2D vB2;
    double angularVelA2(0.0);
    double angularVelB2(0.0);

    Physics::collision(vA1, vB1, rAPp, rBPp, angularVelA1,
                       angularVelB1, n, angularMassA, angularMassB,
                       massA, massB,
                       vA2, vB2, angularVelA2, angularVelB2);
    v2 = (vA2*massA+vB2*massB)/(massA+massB);
    angularVel2 = (angularVelA2*angularMassA+angularVelB2*angularMassB)/angularMassAB;
}


//remember that the coordinate system in a computer system is mirrored at the x-axis...
QVector2D Physics::tangentialVelocity (QVector2D r, QVector2D vel, double angularVel)
{
    return vel - angularVel*rotateQuarterCounterClockwise(r)*degToRad;
}


double Physics::kineticEnergy (double mass, QVector2D vel, double angularMass, double angularVel)
{
	return linearKineticEnergy(mass, vel) + rotationalKineticEnergy(angularMass, angularVel);
}

double Physics::linearKineticEnergy(double mass, QVector2D vel)
{
	return 0.5*mass*vel.lengthSquared();
}

double Physics::rotationalKineticEnergy(double angularMass, double angularVel)
{
	angularVel *= degToRad;
	return 0.5*angularMass*angularVel*angularVel;
}

double Physics::angularMass(vector<QVector2D> const & relPositionOfMasses)
{
	auto result = 0.0;
	for (auto const& relPositionOfMass : relPositionOfMasses) {
		result += relPositionOfMass.lengthSquared();
	}
	return result;
}

double Physics::angularVelocity (double angularMassOld, double angularMassNew, double angularVelOld)
{
    angularVelOld = angularVelOld*degToRad;
    double rotEnergyDouble = angularMassOld*angularVelOld*angularVelOld;
    double angularVelNew = qSqrt(rotEnergyDouble/angularMassNew);
    return angularVelNew*radToDeg;
}

double Physics::angularMomentum (QVector2D r, QVector2D v)
{
    return r.x()*v.y()-r.y()*v.x();     //we only calc the 3rd component of the 3D cross product
}

double Physics::angularVelocity (double angularMomentum, double angularMass)
{
    if( qAbs(angularMass) < FLOATINGPOINT_HIGH_PRECISION )
        return 0;
    else
        return angularMomentum/angularMass*radToDeg;
}

namespace
{
	vector<QVector2D> getPositionsInBarycentricCoordinates(vector<QVector2D> const& positions)
	{
		QVector2D center = std::accumulate(positions.begin(), positions.end(), QVector2D());
		center /= positions.size();
		vector<QVector2D> result;
		std::transform(positions.begin(), positions.end(), std::inserter(result, result.begin()), [&center](QVector2D const& pos) {
			return pos - center;
		});
		return result;
	}
}

auto Physics::velocitiesOfCenter(Velocities const& velocities, vector<QVector2D> const& relPositionOfMasses) -> Velocities
{
	CHECK(!relPositionOfMasses.empty());
	Velocities result;
	result.angular = 0.0;
	for (QVector2D const& relPositionOfMass : relPositionOfMasses) {
		result.linear += Physics::tangentialVelocity(relPositionOfMass, velocities.linear, velocities.angular);
	}
	result.linear /= relPositionOfMasses.size();
	if (relPositionOfMasses.size() == 1) {
		return result;
	}

	double angularMomentum = 0.0;
	double angularMass = Physics::angularMass(getPositionsInBarycentricCoordinates(relPositionOfMasses));
	for (QVector2D const& relPositionOfMass : relPositionOfMasses) {
		QVector2D tangentialVel = Physics::tangentialVelocity(relPositionOfMass, velocities.linear, velocities.angular);
		QVector2D relVel = tangentialVel - result.linear;
		angularMomentum += Physics::angularMomentum(relPositionOfMass, relVel);
	}

	result.angular = Physics::angularVelocity(angularMomentum, angularMass);
	return result;
}

void Physics::applyImpulse (QVector2D const& impulse, QVector2D const& relPos, double mass, QVector2D const& vel, double angularMass, double angularVel, QVector2D& newVel, double& newAngularVel)
{
	QVector2D rAPp = Physics::rotateQuarterCounterClockwise(relPos);
	newVel = vel + impulse / mass;
    newAngularVel = angularVel - QVector2D::dotProduct(rAPp, impulse)/angularMass*radToDeg;
}

QVector2D Physics::rotateClockwise (QVector2D v, double angle)
{
    QMatrix4x4 transform;
    transform.rotate(angle, 0.0, 0.0, 1.0);
    return transform.map(QVector3D(v)).toVector2D();
}

QVector2D Physics::rotateQuarterCounterClockwise (QVector2D v)
{
    //90 degree counterclockwise rotation of vector r
    double temp(v.x());
    v.setX(v.y());
    v.setY(-temp);
    return v;
}

double Physics::angleOfVector (QVector2D v)
{
    double angle(0.0);
    double angleSin(qAsin(-v.y()/v.length())*radToDeg);
    if( v.x() >= 0.0 )
        angle = 90.0-angleSin;
    else
        angle = angleSin+270.0;
    return angle;
}

QVector2D Physics::unitVectorOfAngle (double angle)
{
    return QVector2D(qSin(angle*degToRad),-qCos(angle*degToRad));
}

bool Physics::compareEqualAngle (double angle1, double angle2, double precision)
{
    if( qAbs(angle1-angle2) < precision )
        return true;
    if( qAbs(angle1-angle2-360.0) < precision )
        return true;
    if( qAbs(angle1-angle2+360.0) < precision )
        return true;
    return false;
}

double Physics::subtractAngle (double angleMinuend, double angleSubtrahend)
{
    double angleDiff = angleMinuend- angleSubtrahend;
    if( angleDiff > 360.0 )
        angleDiff -= 360.0;
    if( angleDiff < 0.0 )
        angleDiff += 360.0;
    return angleDiff;
}

double Physics::clockwiseAngleFromFirstToSecondVector (QVector2D v1, QVector2D v2)
{
    double a1 = Physics::angleOfVector(v1);
    double a2 = Physics::angleOfVector(v2);
    return subtractAngle(a2, a1);
}
