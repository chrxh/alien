#pragma once

#include "Array.cuh"
#include "Base.cuh"

// A change to an object that another domain owns. It is sent to the owner in the next sync round and applied there.
enum class DomainOpType : uint8_t
{
    CreditEnergy,
    DrainAttackedEnergy,
    CreditAttacker,
    AddConnection,
    RemoveConnection,
    Inject,
    CommunicatorSignal,
    AddVelocity,
    ActivateDetonator,
    ShockWave,
    ShiftConnectionAngle,
    ChangeConnectionDistance,
    ConfirmCreature,
    ResetMuscle,
};

enum class EnergyKind : uint8_t
{
    Usable,
    Raw,
};

// Aligned so that the values can be read with vector loads
struct __align__(16) DomainOp
{
    DomainOpType type;
    uint8_t targetDomain;
    uint8_t numForwards;
    uint8_t kind;
    uint64_t targetId;
    uint64_t otherId;
    float2 pos;
    float values[10];
};

class DomainOps
{
public:
    static auto constexpr MaxForwards = 4;

    // Returns nullptr if the outbox is full; the caller should then skip the action that would require the op
    __device__ __inline__ static DomainOp* tryCreate(Array<DomainOp>& outbox, DomainOpType type, int targetDomain, uint64_t targetId)
    {
        auto op = outbox.tryGetNewElement();
        if (!op) {
            return nullptr;
        }
        op->type = type;
        op->targetDomain = static_cast<uint8_t>(targetDomain);
        op->numForwards = 0;
        op->kind = 0;
        op->targetId = targetId;
        op->otherId = 0;
        op->pos = {0, 0};
        return op;
    }
};
