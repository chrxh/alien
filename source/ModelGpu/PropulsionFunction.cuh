#pragma once

#include "SimulationData.cuh"
#include "QuantityConverter.cuh"

class PropulsionFunction
{
public:
    __inline__ __device__ static void processing(Token* token, EntityFactory& factory);

private:
    __inline__ __device__ static float convertDataToThrustPower(unsigned char data);

};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void PropulsionFunction::processing(Token * token, EntityFactory& factory)
{
    auto const& cell = token->cell;
    auto const& sourceCell = token->sourceCell;
    auto const& cluster = cell->cluster;
    auto& tokenMem = token->memory;
    auto const& command = static_cast<unsigned char>(tokenMem[Enums::Prop::INPUT]) % Enums::PropIn::_COUNTER;

    if (Enums::PropIn::DO_NOTHING == command) {
        tokenMem[Enums::Prop::OUTPUT] = Enums::PropOut::SUCCESS;
        return;
    }

    float angle = QuantityConverter::convertDataToAngle(tokenMem[Enums::Prop::IN_ANGLE]);
    float power = convertDataToThrustPower(tokenMem[Enums::Prop::IN_POWER]);

    float const clusterMass = cluster->numCellPointers / cudaSimulationParameters.cellMass_Reciprocal;
    auto const& angularVel = cluster->getAngularVelocity_safe();
    auto const& vel = cluster->getVelocity_safe();
    auto const& angularMass = cluster->angularMass;

    auto origKineticEnergy =
        Physics::kineticEnergy(clusterMass, vel, angularMass, angularVel);
    auto cellRelPos = cell->absPos - cluster->pos;
    auto tangVel = Physics::tangentialVelocity(cellRelPos, vel, angularVel);

    //calc angle of acting thrust
    float2 impulse;
    if (Enums::PropIn::BY_ANGLE == command) {
        auto thrustAngle =
            (Math::angleOfVector(sourceCell->relPos - cell->relPos) + cluster->angle + angle);
        impulse = Math::unitVectorOfAngle(thrustAngle) * power;

    }
    if (Enums::PropIn::FROM_CENTER == command) {
        impulse = Math::normalized(cellRelPos) * power;
    }
    if (Enums::PropIn::TOWARD_CENTER == command) {
        impulse = Math::normalized(cellRelPos) * (-power);
    }

    auto rAPp = cellRelPos;
    Math::rotateQuarterCounterClockwise(rAPp);
    Math::normalize(rAPp);
    if (Enums::PropIn::ROTATION_CLOCKWISE == command) {
        impulse = rAPp * (-power);
    }
    if (Enums::PropIn::ROTATION_COUNTERCLOCKWISE == command) {
        impulse = rAPp * power;
    }
    if (Enums::PropIn::DAMP_ROTATION == command) {
        if (angularVel > 0.0f) {
            impulse = rAPp * power;
        }
        else if (angularVel < 0.0f) {
            impulse = rAPp * (-power);
        }
        else {
            impulse = {0, 0};
        }
    }

    //calc impact of impulse to cell structure
    float2 velInc;
    float angularVelInc;
    Physics::calcImpulseIncrement(impulse, cellRelPos, clusterMass, angularMass, velInc, angularVelInc);

    //only for damping: prove if its too much => do nothing
    if (Enums::PropIn::DAMP_ROTATION == command) {
        if ((angularVel >= 0.0 && angularVel + angularVelInc <= 0.0) ||
            (angularVel <= 0.0 && angularVel + angularVelInc >= 0.0)) {

            tokenMem[Enums::Prop::OUTPUT] = Enums::PropOut::SUCCESS_DAMPING_FINISHED;
            return;
        }
    }

    auto newKineticEnergy =
        Physics::kineticEnergy(clusterMass, vel + velInc, angularMass, angularVel + angularVelInc);
    auto energyDiff = newKineticEnergy - origKineticEnergy;

    if (energyDiff > 0.0f && token->getEnergy() < 2*energyDiff + cudaSimulationParameters.tokenMinEnergy) {
        tokenMem[Enums::Prop::OUTPUT] = Enums::PropOut::ERROR_NO_ENERGY;
        return;
    }

    cluster->addVelocity_safe(velInc);
    cluster->addAngularVelocity_safe(angularVelInc);

    //create energy particle with difference energy
    Math::normalize(impulse);
    auto particlePos = cell->absPos - impulse;
    auto particleVel = tangVel - impulse / 4.0f;
    factory.createParticle(abs(energyDiff), particlePos, particleVel, { cell->metadata.color });

    token->changeEnergy(-(energyDiff + abs(energyDiff)));
    tokenMem[Enums::Prop::OUTPUT] = Enums::PropOut::SUCCESS;

}

__inline__ __device__ float PropulsionFunction::convertDataToThrustPower(unsigned char data)
{
    return 1.0f / 10000.0f*(static_cast<float>(data) + 10.0f);
}
 