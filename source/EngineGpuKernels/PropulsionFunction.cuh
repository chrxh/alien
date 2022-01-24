#pragma once

#include "SimulationData.cuh"
#include "QuantityConverter.cuh"

class PropulsionFunction
{
public:
    __inline__ __device__ static void processing(Token* token, SimulationData& data);

private:
    __inline__ __device__ static float convertDataToThrustPower(unsigned char data);

};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void PropulsionFunction::processing(Token* token, SimulationData& data)
{
    auto& tokenMem = token->memory;

/*
    auto const& cell = token->cell;
    auto const& sourceCell = token->sourceCell;
    auto const& command = static_cast<unsigned char>(tokenMem[Enums::Prop::INPUT]) % Enums::PropIn::_COUNTER;

    if (Enums::PropIn::DO_NOTHING == command) {
        tokenMem[Enums::Prop_Output] = Enums::PropOut_Success;
        return;
    }
    float power = convertDataToThrustPower(tokenMem[Enums::Prop::IN_POWER]);
    auto energyCost = power / 100;
    if (token->energy < energyCost + cudaSimulationParameters.tokenMinEnergy) {
        tokenMem[Enums::Prop_Output] = Enums::PropOut::ERROR_NO_ENERGY;
        return;
    }

    float2 impulse;
    if (Enums::PropIn::BY_ANGLE == command) {
        float angle = QuantityConverter::convertDataToAngle(tokenMem[Enums::Prop::IN_ANGLE]);

        auto posDelta = cell->absPos - sourceCell->absPos;
        data.cellMap.mapDisplacementCorrection(posDelta);
        auto thrustAngle = Math::angleOfVector(posDelta) + angle;
        impulse = Math::unitVectorOfAngle(thrustAngle) * power;
    }
    if (Enums::PropIn::DAMP_ROTATION == command) {
        auto rotVel = Math::normalized(cell->vel) - Math::normalized(cell->temp3);
        cell->temp3 = cell->vel;
        impulse = Math::normalized(rotVel) * (-power);
    }

    atomicAdd(&cell->vel.x, impulse.x);
    atomicAdd(&cell->vel.y, impulse.y);

    Math::normalize(impulse);
    auto particlePos = cell->absPos - impulse;
    auto particleVel = cell->vel - impulse / 4.0f;

    EntityFactory factory;
    factory.init(&data);
    factory.createParticle(energyCost, particlePos, particleVel, {cell->metadata.color});

    token->energy -= energyCost;
*/
    tokenMem[Enums::Prop_Output] = Enums::PropOut_Success;
}

__inline__ __device__ float PropulsionFunction::convertDataToThrustPower(unsigned char data)
{
    return 1.0f / 1000.0f*(static_cast<float>(data) + 10.0f);
}
 