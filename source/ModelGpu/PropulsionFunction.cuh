#pragma once

#include "SimulationData.cuh"
#include "QuantityConverter.cuh"

class PropulsionFunction
{
public:
    __inline__ __device__ static void processing(Cell const* sourceCell, Token* token, EntityFactory& factory);

private:
    __inline__ __device__ static float convertDataToThrustPower(unsigned char data);

};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void PropulsionFunction::processing(Cell const* sourceCell, Token * token, EntityFactory& factory)
{
    auto const& cell = token->cell;
    auto const& cluster = cell->cluster;
    auto& tokenMem = token->memory;
    auto const& command = tokenMem[Enums::Prop::IN] % Enums::PropIn::_COUNTER;

    float angle = QuantityConverter::convertDataToAngle(tokenMem[Enums::Prop::IN_ANGLE]);
    float power = convertDataToThrustPower(tokenMem[Enums::Prop::IN_POWER]);

    if (Enums::PropIn::DO_NOTHING == command) {
        tokenMem[Enums::Prop::OUT] = Enums::PropOut::SUCCESS;
        return;
    }

    int locked;
    do {    //mutex
        locked = atomicExch(&cluster->locked, 1);
        if (0 == locked) {
            auto clusterMass = cluster->numCells * cudaSimulationParameters.cellMass;
            auto const& angularVel = cluster->angularVel;
            auto const& vel = cluster->vel;
            auto const& angularMass = cluster->angularMass;

            auto origKineticEnergy =
                Physics::kineticEnergy(clusterMass, vel, angularMass, angularVel);
            auto cellRelPos = Math::sub(cell->absPos, cluster->pos);
            auto tangVel = Physics::tangentialVelocity(cellRelPos, vel, angularVel);

            //calc angle of acting thrust
            float2 impulse;
            if (Enums::PropIn::BY_ANGLE == command) {
                auto thrustAngle =
                    (Math::angleOfVector(Math::sub(sourceCell->relPos, cell->relPos)) + cluster->angle + angle);
                impulse = Math::mul(Math::unitVectorOfAngle(thrustAngle), power);
            }
            if (Enums::PropIn::FROM_CENTER == command) {
                impulse = Math::mul(Math::normalized(cellRelPos), power);
            }
            if (Enums::PropIn::TOWARD_CENTER == command) {
                impulse = Math::mul(Math::normalized(Math::minus(cellRelPos)), power);
            }

            auto rAPp = cellRelPos;
            Math::rotateQuarterCounterClockwise(rAPp);
            Math::normalize(rAPp);
            if (Enums::PropIn::ROTATION_CLOCKWISE == command) {
                impulse = Math::mul(rAPp, -power);
            }
            if (Enums::PropIn::ROTATION_COUNTERCLOCKWISE == command) {
                impulse = Math::mul(rAPp, power);
            }
            if (Enums::PropIn::DAMP_ROTATION == command) {
                if (angularVel > 0.0f) {
                    impulse = Math::mul(rAPp, power);
                }
                if (angularVel < 0.0f) {
                    impulse = Math::mul(rAPp, -power);
                }
            }

            //calc impact of impulse to cell structure
            float2 velInc;
            float angularVelInc;
            Physics::calcImpulseIncrement(impulse, cellRelPos, clusterMass, angularMass, velInc, angularVelInc);

            //only for damping: prove if its too much => do nothing
            if (Enums::PropIn::DAMP_ROTATION == command) {
                if ((angularVel > 0.0 && angularVel + angularVelInc < 0.0) ||
                    (angularVel < 0.0 && angularVel + angularVelInc > 0.0)) {

                    tokenMem[Enums::Prop::OUT] = Enums::PropOut::SUCCESS_DAMPING_FINISHED;
                    atomicExch(&cluster->locked, 0);
                    return;
                }
            }

            auto newKineticEnergy =
                Physics::kineticEnergy(clusterMass, Math::add(vel, velInc), angularMass, angularVel + angularVelInc);
            auto energyDiff = newKineticEnergy - origKineticEnergy;

            if (energyDiff > 0.0f && token->energy < energyDiff + cudaSimulationParameters.tokenMinEnergy + FP_PRECISION) {
                tokenMem[Enums::Prop::OUT] = Enums::PropOut::ERROR_NO_ENERGY;
                atomicExch(&cluster->locked, 0);
                return;
            }

            cluster->vel = Math::add(vel, velInc);
            cluster->angularVel += angularVelInc;

            //create energy particle with difference energy
            Math::normalize(impulse);
            auto particlePos = Math::sub(cell->absPos, impulse);
            auto particleVel = Math::sub(tangVel, Math::div(impulse, 4.0f));
            factory.createParticle(abs(energyDiff), particlePos, particleVel);

            token->energy -= (energyDiff + abs(energyDiff));
            tokenMem[Enums::Prop::OUT] = Enums::PropOut::SUCCESS;

            atomicExch(&cluster->locked, 0);
        }
    } while (1 == locked);
}

__inline__ __device__ float PropulsionFunction::convertDataToThrustPower(unsigned char data)
{
    return 1.0f / 10000.0f*(static_cast<float>(data) + 10.0f);
}
