#pragma once

#include "DomainOps.cuh"
#include "SimulationData.cuh"

// Creates the ops for changes to ghosts. The owner of a ghost applies them in the next sync round, see DomainOpProcessor.
// The functions return false if the outbox is full; the caller then has to skip its own part of the change.
class DomainOpEmitter
{
public:
    __inline__ __device__ static bool creditEnergy(SimulationData& data, Object* target, EnergyKind kind, float amount);
    __inline__ __device__ static bool drainAttackedEnergy(SimulationData& data, Object* target, Object* attacker, float amount);
    __inline__ __device__ static bool addConnection(SimulationData& data, Object* ghost, Object* ownObject, float distance);
    __inline__ __device__ static bool removeConnection(SimulationData& data, Object* ghost, Object* ownObject);
    __inline__ __device__ static bool inject(SimulationData& data, Object* target, Creature* injectorCreature, int geneIndex);
    __inline__ __device__ static bool communicatorSignal(SimulationData& data, Object* receiver, float const* signals, float2 const& senderFacing);
    __inline__ __device__ static bool addVelocity(SimulationData& data, Object* target, float2 const& velocityDelta);
    __inline__ __device__ static bool activateDetonator(SimulationData& data, Object* target);
    __inline__ __device__ static void shockWave(SimulationData& data, float2 const& center, float innerRadius, float outerRadius, float radius, int detached);
    __inline__ __device__ static bool shiftConnectionAngle(SimulationData& data, Object* pivot, Object* muscle, float angleDelta);
    __inline__ __device__ static bool changeConnectionDistance(SimulationData& data, Object* target, Object* connectedObject, float distanceDelta);
    __inline__ __device__ static bool confirmCreature(SimulationData& data, Object* cellOfCreature);
    __inline__ __device__ static bool resetMuscle(SimulationData& data, Object* muscle, bool resetFrontAngle, bool restoreInitialAngle);

    __inline__ __device__ static float asFloat(uint32_t value) { return __uint_as_float(value); }
    __inline__ __device__ static uint32_t asUInt32(float value) { return __float_as_uint(value); }

private:
    __inline__ __device__ static DomainOp* createFor(SimulationData& data, DomainOpType type, Object* target);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ DomainOp* DomainOpEmitter::createFor(SimulationData& data, DomainOpType type, Object* target)
{
    auto op = DomainOps::tryCreate(data.domainOps, type, target->ownerDomain, target->id);
    if (op) {
        op->pos = target->pos;
    }
    return op;
}

__inline__ __device__ bool DomainOpEmitter::creditEnergy(SimulationData& data, Object* target, EnergyKind kind, float amount)
{
    auto op = createFor(data, DomainOpType::CreditEnergy, target);
    if (!op) {
        return false;
    }
    op->kind = static_cast<uint8_t>(kind);
    op->values[0] = amount;
    return true;
}

__inline__ __device__ bool DomainOpEmitter::drainAttackedEnergy(SimulationData& data, Object* target, Object* attacker, float amount)
{
    auto op = createFor(data, DomainOpType::DrainAttackedEnergy, target);
    if (!op) {
        return false;
    }
    op->otherId = attacker->id;
    op->kind = static_cast<uint8_t>(data.domain.index);
    op->values[0] = amount;
    op->values[1] = asFloat(attacker->typeData.cell.creature->lineageId);
    op->values[2] = attacker->pos.x;
    op->values[3] = attacker->pos.y;
    return true;
}

__inline__ __device__ bool DomainOpEmitter::addConnection(SimulationData& data, Object* ghost, Object* ownObject, float distance)
{
    auto op = createFor(data, DomainOpType::AddConnection, ghost);
    if (!op) {
        return false;
    }
    op->otherId = ownObject->id;
    op->kind = static_cast<uint8_t>(data.domain.index);
    op->values[0] = distance;
    return true;
}

__inline__ __device__ bool DomainOpEmitter::removeConnection(SimulationData& data, Object* ghost, Object* ownObject)
{
    auto op = createFor(data, DomainOpType::RemoveConnection, ghost);
    if (!op) {
        return false;
    }
    op->otherId = ownObject->id;
    return true;
}

__inline__ __device__ bool DomainOpEmitter::inject(SimulationData& data, Object* target, Creature* injectorCreature, int geneIndex)
{
    auto op = createFor(data, DomainOpType::Inject, target);
    if (!op) {
        return false;
    }
    op->otherId = injectorCreature->id;
    op->values[0] = toFloat(geneIndex);
    return true;
}

__inline__ __device__ bool DomainOpEmitter::communicatorSignal(SimulationData& data, Object* receiver, float const* signals, float2 const& senderFacing)
{
    static_assert(STANDARD_NEURONS_PER_CELL + 2 <= sizeof(DomainOp::values) / sizeof(float));
    auto op = createFor(data, DomainOpType::CommunicatorSignal, receiver);
    if (!op) {
        return false;
    }
    for (int i = 0; i < STANDARD_NEURONS_PER_CELL; ++i) {
        op->values[i] = signals[i];
    }
    op->values[STANDARD_NEURONS_PER_CELL] = senderFacing.x;
    op->values[STANDARD_NEURONS_PER_CELL + 1] = senderFacing.y;
    return true;
}

__inline__ __device__ bool DomainOpEmitter::addVelocity(SimulationData& data, Object* target, float2 const& velocityDelta)
{
    auto op = createFor(data, DomainOpType::AddVelocity, target);
    if (!op) {
        return false;
    }
    op->values[0] = velocityDelta.x;
    op->values[1] = velocityDelta.y;
    return true;
}

__inline__ __device__ bool DomainOpEmitter::activateDetonator(SimulationData& data, Object* target)
{
    return createFor(data, DomainOpType::ActivateDetonator, target) != nullptr;
}

// The front of a shock wave also passes the objects of the other domains, which apply it to their own objects
__inline__ __device__ void
DomainOpEmitter::shockWave(SimulationData& data, float2 const& center, float innerRadius, float outerRadius, float radius, int detached)
{
    for (int domain = 0; domain < data.domain.numDomains; ++domain) {
        if (domain == data.domain.index) {
            continue;
        }
        auto op = DomainOps::tryCreate(data.domainOps, DomainOpType::ShockWave, domain, 0);
        if (!op) {
            return;
        }
        op->pos = center;
        op->kind = static_cast<uint8_t>(detached);
        op->values[0] = innerRadius;
        op->values[1] = outerRadius;
        op->values[2] = radius;
    }
}

__inline__ __device__ bool DomainOpEmitter::shiftConnectionAngle(SimulationData& data, Object* pivot, Object* muscle, float angleDelta)
{
    auto op = createFor(data, DomainOpType::ShiftConnectionAngle, pivot);
    if (!op) {
        return false;
    }
    op->otherId = muscle->id;
    op->values[0] = angleDelta;
    return true;
}

__inline__ __device__ bool DomainOpEmitter::changeConnectionDistance(SimulationData& data, Object* target, Object* connectedObject, float distanceDelta)
{
    auto op = createFor(data, DomainOpType::ChangeConnectionDistance, target);
    if (!op) {
        return false;
    }
    op->otherId = connectedObject->id;
    op->values[0] = distanceDelta;
    return true;
}

__inline__ __device__ bool DomainOpEmitter::confirmCreature(SimulationData& data, Object* cellOfCreature)
{
    return createFor(data, DomainOpType::ConfirmCreature, cellOfCreature) != nullptr;
}

__inline__ __device__ bool DomainOpEmitter::resetMuscle(SimulationData& data, Object* muscle, bool resetFrontAngle, bool restoreInitialAngle)
{
    auto op = createFor(data, DomainOpType::ResetMuscle, muscle);
    if (!op) {
        return false;
    }
    op->kind = (resetFrontAngle ? 1 : 0) | (restoreInitialAngle ? 2 : 0);
    return true;
}
