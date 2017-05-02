#include <QtCore/qmath.h>

#include "global/servicelocator.h"
#include "model/entities/cell.h"
#include "model/entities/cellcluster.h"
#include "model/entities/energyparticle.h"
#include "model/entities/token.h"
#include "model/entities/entityfactory.h"
#include "model/physics/physics.h"
#include "model/physics/codingphysicalquantities.h"
#include "model/modelsettings.h"
#include "model/context/unitcontext.h"
#include "model/context/simulationparameters.h"

#include "cellfunctionpropulsionimpl.h"


CellFunctionPropulsionImpl::CellFunctionPropulsionImpl (UnitContext* context)
    : CellFunction(context)
{
}

namespace {
    qreal convertDataToThrustPower (quint8 b)
    {
        return 1/10000.0*((qreal)b+10.0);
    }
}

CellFeature::ProcessingResult CellFunctionPropulsionImpl::processImpl (Token* token, Cell* cell, Cell* previousCell)
{
    ProcessingResult processingResult {false, 0};
    CellCluster* cluster(cell->getCluster());
	auto& tokenMem = token->getMemoryRef();
    quint8 cmd = tokenMem[Enums::Prop::IN] % 7;
    qreal angle = CodingPhysicalQuantities::convertDataToAngle(tokenMem[Enums::Prop::IN_ANGLE]);
    qreal power = convertDataToThrustPower(tokenMem[Enums::Prop::IN_POWER]);

    if( cmd == Enums::PropIn::DO_NOTHING ) {
        tokenMem[Enums::Prop::OUT] = Enums::PropOut::SUCCESS;
        return processingResult;
    }

    //calc old kinetic energy
    qreal eKinOld(Physics::kineticEnergy(cluster->getMass(), cluster->getVelocity(), cluster->getAngularMass(), cluster->getAngularVel()));

    //calc old tangential velocity
    QVector3D cellRelPos(cluster->calcPosition(cell)-cluster->getPosition());
    QVector3D tangVel(Physics::tangentialVelocity(cellRelPos, cluster->getVelocity(), cluster->getAngularVel()));

    //calc impulse angle
    QVector3D impulse(0.0, 0.0, 0.0);
    if( cmd == Enums::PropIn::BY_ANGLE ) {
        qreal thrustAngle = (Physics::angleOfVector(-cell->getRelPosition() + previousCell->getRelPosition())+cluster->getAngle()+ angle)*degToRad;
        impulse = QVector3D(qSin(thrustAngle), -qCos(thrustAngle), 0.0)*power;
    }
    if( cmd == Enums::PropIn::FROM_CENTER ) {
        impulse = cellRelPos.normalized()*power;
    }
    if( cmd == Enums::PropIn::TOWARD_CENTER ) {
        impulse = -cellRelPos.normalized()*power;
    }

    QVector3D rAPp = cellRelPos;
    rAPp = Physics::rotateQuarterCounterClockwise(rAPp);
    if( cmd == Enums::PropIn::ROTATION_CLOCKWISE ) {
        impulse = -rAPp.normalized()*power;
    }
    if( cmd == Enums::PropIn::ROTATION_COUNTERCLOCKWISE ) {
        impulse = rAPp.normalized()*power;
    }
    if( cmd == Enums::PropIn::DAMP_ROTATION ) {
        if( cluster->getAngularVel() > 0.00 )
            impulse = rAPp.normalized()*power;
        if( cluster->getAngularVel() < 0.00 )
            impulse = -rAPp.normalized()*power;
    }

    //calc impact of impulse to cell structure
    QVector3D newVel;
    qreal newAngularVel;
    Physics::applyImpulse(impulse, rAPp, cluster->getMass(), cluster->getVelocity(), cluster->getAngularMass(), cluster->getAngularVel(), newVel, newAngularVel);

    //only for damping: prove if its too much
    if( cmd == Enums::PropIn::DAMP_ROTATION ) {
        if( (cluster->getAngularVel() > 0.0 && newAngularVel < 0.0)
                || (cluster->getAngularVel() < 0.0 && newAngularVel > 0.0) ) {
            newVel = cluster->getVelocity();
            newAngularVel = cluster->getAngularVel();

            //update return value
            tokenMem[Enums::Prop::OUT] = Enums::PropOut::SUCCESS_DAMPING_FINISHED;
            return processingResult;
        }
    }

    //calc new kinetic energy
	auto parameters = _context->getSimulationParameters();
    qreal eKinNew(Physics::kineticEnergy(cluster->getMass(), newVel, cluster->getAngularMass(), newAngularVel));
    qreal energyDiff((eKinNew-eKinOld)/ parameters->cellMass_Reciprocal);

    //has token enough energy?
    if( token->getEnergy() >= (energyDiff + qAbs(energyDiff) + parameters->tokenMinEnergy + ALIEN_PRECISION) ) {

        //create energy particle with difference energy
		auto factory = ServiceLocator::getInstance().getService<EntityFactory>();
		processingResult.newEnergyParticle = factory->buildEnergyParticle(qAbs(energyDiff)
			, cluster->calcPosition(cell, _context) - impulse.normalized()
			, tangVel - impulse.normalized() / 4.0, _context);

        //update velocities
        cluster->setVelocity(newVel);
        cluster->setAngularVel(newAngularVel);
        token->setEnergy(token->getEnergy() - (energyDiff+qAbs(energyDiff)));

        //update return value
        tokenMem[Enums::Prop::OUT] = Enums::PropOut::SUCCESS;
    }
    else {

        //update return value
        tokenMem[Enums::Prop::OUT] = Enums::PropOut::ERROR_NO_ENERGY;
    }
    return processingResult;
}

