#pragma once

#include <Base/Definitions.h>
#include <Base/Singleton.h>

#include "Descs.h"

class DescEditService
{
    MAKE_SINGLETON(DescEditService);

public:
    void scaleContent(ContentDesc& description, IntVector2D const& origWorldSize, IntVector2D const& worldSize) const;

    using Occupancy = std::unordered_map<IntVector2D, std::vector<RealVector2D>>;
    void addIfSpaceAvailable(ContentDesc& result, Occupancy& occupancy, ContentDesc const& toAdd, float distance, IntVector2D const& worldSize) const;
    bool isCellPresent(Occupancy const& cellPosBySlot, SpaceCalculator const& spaceCalculator, RealVector2D const& posToCheck, float distance) const;

    void flattenTopology(ContentDesc& description, IntVector2D const& worldSize) const;

    void reconnectObjects(ContentDesc& description, float maxDistance) const;  // For non-creatures

    uint64_t getId(ExtendedObjectOrEnergyDesc const& entity) const;
    RealVector2D getPos(ExtendedObjectOrEnergyDesc const& entity) const;
    std::vector<ExtendedObjectOrEnergyDesc> getObjects(ContentDesc const& description) const;

    void setCenter(ContentDesc& collection, RealVector2D const& center) const;
    RealVector2D calcCenter(ContentDesc const& collection) const;
    void shift(ContentDesc& collection, RealVector2D const& delta) const;
    void rotate(ContentDesc& collection, float angle) const;
    void accelerate(ContentDesc& collection, RealVector2D const& velDelta, float angularVelDelta) const;

    void removeCell(ContentDesc& collection, uint64_t objectId) const;
    void removeCellIf(ContentDesc& collection, std::function<bool(ObjectDesc const&)> const& predicate) const;
};
