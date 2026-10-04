#pragma once

#include "AttackerProcessor.cuh"
#include "CommunicatorProcessor.cuh"
#include "DetonatorProcessor.cuh"
#include "DomainOpEmitter.cuh"
#include "DomainSync.cuh"
#include "InjectorProcessor.cuh"
#include "MuscleProcessor.cuh"
#include "ObjectConnectionProcessor.cuh"
#include "SimulationStatistics.cuh"

// Applies the ops that other domains sent for the own objects
class DomainOpProcessor
{
public:
    __inline__ __device__ static void apply(SimulationData& data, SimulationStatistics& statistics, DomainSyncData const& syncData, DomainOp const& op);

private:
    __inline__ __device__ static void forward(SimulationData& data, DomainOp const& op, int owner);
    __inline__ __device__ static void handleMissingTarget(SimulationData& data, DomainOp const& op);
    __inline__ __device__ static void createEnergyParticle(SimulationData& data, float2 const& pos, float energy);

    __inline__ __device__ static void creditEnergy(SimulationData& data, Object* target, DomainOp const& op);
    __inline__ __device__ static void drainAttackedEnergy(SimulationData& data, Object* target, DomainOp const& op);
    __inline__ __device__ static void creditAttacker(SimulationData& data, SimulationStatistics& statistics, Object* target, DomainOp const& op);
    __inline__ __device__ static void addConnection(SimulationData& data, DomainSyncData const& syncData, Object* target, DomainOp const& op);
    __inline__ __device__ static void removeConnection(DomainSyncData const& syncData, Object* target, DomainOp const& op);
    __inline__ __device__ static void inject(SimulationData& data, DomainSyncData const& syncData, Object* target, DomainOp const& op);
    __inline__ __device__ static void shiftConnectionAngle(DomainSyncData const& syncData, Object* target, DomainOp const& op);
    __inline__ __device__ static void changeConnectionDistance(DomainSyncData const& syncData, Object* target, DomainOp const& op);
    __inline__ __device__ static void resetMuscle(Object* target, DomainOp const& op);

    __inline__ __device__ static int findConnectionIndex(Object* object, Object* connectedObject);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void DomainOpProcessor::apply(SimulationData& data, SimulationStatistics& statistics, DomainSyncData const& syncData, DomainOp const& op)
{
    if (op.type == DomainOpType::ShockWave) {
        if (auto shockWave = data.receivedShockWaves.tryGetNewElement()) {
            *shockWave = op;
        }
        return;
    }
    auto target = syncData.objectMap.find(op.targetId);
    if (target && target->isRemovedGhost()) {
        target = nullptr;
    }
    if (!target) {
        handleMissingTarget(data, op);
        return;
    }
    if (target->isGhost()) {
        forward(data, op, target->ownerDomain);
        return;
    }

    switch (op.type) {
    case DomainOpType::CreditEnergy:
        creditEnergy(data, target, op);
        break;
    case DomainOpType::DrainAttackedEnergy:
        drainAttackedEnergy(data, target, op);
        break;
    case DomainOpType::CreditAttacker:
        creditAttacker(data, statistics, target, op);
        break;
    case DomainOpType::AddConnection:
        addConnection(data, syncData, target, op);
        break;
    case DomainOpType::RemoveConnection:
        removeConnection(syncData, target, op);
        break;
    case DomainOpType::Inject:
        inject(data, syncData, target, op);
        break;
    case DomainOpType::CommunicatorSignal:
        if (target->type == ObjectType_Cell && target->typeData.cell.frontAngle != VALUE_NOT_SET_FLOAT) {
            CommunicatorProcessor::receiveSignal(data, target, op.values, {op.values[STANDARD_NEURONS_PER_CELL], op.values[STANDARD_NEURONS_PER_CELL + 1]});
        }
        break;
    case DomainOpType::AddVelocity:
        atomicAdd(&target->vel.x, op.values[0]);
        atomicAdd(&target->vel.y, op.values[1]);
        break;
    case DomainOpType::ActivateDetonator:
        if (target->type == ObjectType_Cell && target->typeData.cell.cellType == CellType_Detonator) {
            auto& detonator = target->typeData.cell.cellTypeData.detonator;
            if (detonator.state != DetonatorState_Exploded) {
                detonator.state = DetonatorState_Activated;
                detonator.countdown = 1;
            }
        }
        break;
    case DomainOpType::ShiftConnectionAngle:
        shiftConnectionAngle(syncData, target, op);
        break;
    case DomainOpType::ChangeConnectionDistance:
        changeConnectionDistance(syncData, target, op);
        break;
    case DomainOpType::ConfirmCreature:
        if (target->type == ObjectType_Cell) {
            target->typeData.cell.creature->creatureState = CreatureState_HostConfirmed;
        }
        break;
    case DomainOpType::ResetMuscle:
        resetMuscle(target, op);
        break;
    default:
        break;
    }
}

// The target changed its owner after the op was sent
__inline__ __device__ void DomainOpProcessor::forward(SimulationData& data, DomainOp const& op, int owner)
{
    if (op.numForwards < DomainOps::MaxForwards && owner != data.domain.index) {
        if (auto forwardedOp = DomainOps::tryCreate(data.domainOps, op.type, owner, op.targetId)) {
            *forwardedOp = op;
            forwardedOp->targetDomain = static_cast<uint8_t>(owner);
            forwardedOp->numForwards = op.numForwards + 1;
            return;
        }
    }
    handleMissingTarget(data, op);
}

// Energy must not get lost: it is released as an energy particle where the target was last seen
__inline__ __device__ void DomainOpProcessor::handleMissingTarget(SimulationData& data, DomainOp const& op)
{
    if (op.type == DomainOpType::CreditEnergy || op.type == DomainOpType::CreditAttacker) {
        createEnergyParticle(data, op.pos, op.values[0]);
    } else if (op.type == DomainOpType::AddConnection) {
        if (auto reply = DomainOps::tryCreate(data.domainOps, DomainOpType::RemoveConnection, op.kind, op.otherId)) {
            reply->otherId = op.targetId;
        }
    }
}

__inline__ __device__ void DomainOpProcessor::createEnergyParticle(SimulationData& data, float2 const& pos, float energy)
{
    if (energy <= NEAR_ZERO) {
        return;
    }
    EntityFactory factory;
    factory.init(&data);
    auto correctedPos = pos;
    data.world.correctPosition(correctedPos);
    factory.createEnergy(energy, correctedPos, {0, 0}, 0);
}

__inline__ __device__ void DomainOpProcessor::creditEnergy(SimulationData& data, Object* target, DomainOp const& op)
{
    auto energy = op.values[0];
    switch (target->type) {
    case ObjectType_Cell:
        atomicAdd(static_cast<EnergyKind>(op.kind) == EnergyKind::Raw ? &target->typeData.cell.rawEnergy : &target->typeData.cell.usableEnergy, energy);
        break;
    case ObjectType_FreeCell:
        atomicAdd(&target->typeData.freeCell.energy, energy);
        break;
    case ObjectType_Solid:
        atomicAdd(&target->typeData.solid.energy, energy);
        break;
    case ObjectType_Fluid:
        atomicAdd(&target->typeData.fluid.energy, energy);
        break;
    default:
        createEnergyParticle(data, target->pos, energy);
        break;
    }
}

__inline__ __device__ void DomainOpProcessor::drainAttackedEnergy(SimulationData& data, Object* target, DomainOp const& op)
{
    auto requestedEnergy = op.values[0];
    float2 attackerPos{op.values[2], op.values[3]};

    auto drainedEnergy = 0.0f;
    if (target->type == ObjectType_FreeCell) {
        auto& freeCell = target->typeData.freeCell;
        freeCell.event = CellEvent_Attacked;
        freeCell.eventCounter = 10;
        freeCell.eventPos = attackerPos;
        auto origEnergy = atomicAdd(&freeCell.energy, -requestedEnergy);
        if (origEnergy > requestedEnergy) {
            drainedEnergy = requestedEnergy;
        } else {
            atomicAdd(&freeCell.energy, requestedEnergy);
        }
    } else if (target->type == ObjectType_Cell) {
        auto& cell = target->typeData.cell;
        cell.event = CellEvent_Attacked;
        cell.eventCounter = 10;
        cell.eventPos = attackerPos;
        drainedEnergy = AttackerProcessor::absorbAttackableEnergy(&cell, requestedEnergy);
    }
    if (drainedEnergy <= NEAR_ZERO) {
        return;
    }

    auto reply = DomainOps::tryCreate(data.domainOps, DomainOpType::CreditAttacker, op.kind, op.otherId);
    if (!reply) {
        createEnergyParticle(data, target->pos, drainedEnergy);
        return;
    }
    reply->pos = attackerPos;
    reply->values[0] = drainedEnergy;
    reply->values[1] = op.values[1];
}

__inline__ __device__ void DomainOpProcessor::creditAttacker(SimulationData& data, SimulationStatistics& statistics, Object* target, DomainOp const& op)
{
    auto energy = op.values[0];
    if (target->type != ObjectType_Cell) {
        createEnergyParticle(data, target->pos, energy);
        return;
    }
    auto& cell = target->typeData.cell;
    atomicAdd(&cell.rawEnergy, energy);
    statistics.addAttackedEnergy(DomainOpEmitter::asUInt32(op.values[1]), energy);
    cell.event = CellEvent_Attacking;
    cell.eventCounter = 6;

    // The local part of the attack has already set the success signal
    auto& successSignal = cell.neuralActivity.signals[Channels::AttackerSuccess];
    successSignal = min(1.0f, max(0.0f, successSignal + energy / 10));
}

__inline__ __device__ void DomainOpProcessor::addConnection(SimulationData& data, DomainSyncData const& syncData, Object* target, DomainOp const& op)
{
    auto requester = syncData.objectMap.find(op.otherId);
    auto success = false;
    if (requester && !requester->isRemovedGhost()) {
        target->getLock();
        auto alreadyConnected = findConnectionIndex(target, requester) >= 0;
        success = alreadyConnected;
        if (!alreadyConnected && target->numConnections < MAX_OBJECT_CONNECTIONS) {
            success = ObjectConnectionProcessor::tryAddConnectionOneWay(data, target, requester, op.values[0]);
        }
        target->releaseLock();
    }
    if (!success) {
        if (auto reply = DomainOps::tryCreate(data.domainOps, DomainOpType::RemoveConnection, op.kind, op.otherId)) {
            reply->otherId = op.targetId;
        }
    }
}

__inline__ __device__ void DomainOpProcessor::removeConnection(DomainSyncData const& syncData, Object* target, DomainOp const& op)
{
    auto connectedObject = syncData.objectMap.find(op.otherId);
    if (!connectedObject) {
        return;
    }
    target->getLock();
    ObjectConnectionProcessor::deleteConnectionOneWay(target, connectedObject);
    target->releaseLock();
}

__inline__ __device__ void DomainOpProcessor::inject(SimulationData& data, DomainSyncData const& syncData, Object* target, DomainOp const& op)
{
    if (target->type != ObjectType_Cell || !target->typeData.cell.constructorAvailable || target->typeData.cell.creature->genome->resistanceToInjection) {
        return;
    }
    auto injectorCreature = syncData.creatureMap.find(op.otherId);
    if (!injectorCreature || injectorCreature->genome->isPlaceholder) {
        return;
    }
    InjectorProcessor::injectIntoCell(data, target, injectorCreature, toInt(op.values[0]));
}

__inline__ __device__ void DomainOpProcessor::shiftConnectionAngle(DomainSyncData const& syncData, Object* target, DomainOp const& op)
{
    auto muscle = syncData.objectMap.find(op.otherId);
    if (!muscle || target->numConnections == 0) {
        return;
    }
    auto index = findConnectionIndex(target, muscle);
    if (index < 0) {
        return;
    }
    atomicAdd(&target->connections[index].angleFromPrevious, op.values[0]);
    atomicAdd(&target->connections[(index + 1) % target->numConnections].angleFromPrevious, -op.values[0]);
}

__inline__ __device__ void DomainOpProcessor::changeConnectionDistance(DomainSyncData const& syncData, Object* target, DomainOp const& op)
{
    auto connectedObject = syncData.objectMap.find(op.otherId);
    if (!connectedObject) {
        return;
    }
    auto index = findConnectionIndex(target, connectedObject);
    if (index >= 0) {
        atomicAdd(&target->connections[index].distance, op.values[0]);
    }
}

__inline__ __device__ void DomainOpProcessor::resetMuscle(Object* target, DomainOp const& op)
{
    if (target->type != ObjectType_Cell || target->typeData.cell.cellType != CellType_Muscle) {
        return;
    }
    if (op.kind & 1) {
        target->typeData.cell.frontAngle = VALUE_NOT_SET_FLOAT;
    }
    if ((op.kind & 2) && target->numConnections > 0) {
        auto pivotObject = target->connections[0].object;
        pivotObject->getLock();
        MuscleProcessor::restoreInitialAngleFromPrevious(target);
        pivotObject->releaseLock();
    }
}

__inline__ __device__ int DomainOpProcessor::findConnectionIndex(Object* object, Object* connectedObject)
{
    for (int i = 0; i < object->numConnections; ++i) {
        if (object->connections[i].object == connectedObject) {
            return i;
        }
    }
    return -1;
}
