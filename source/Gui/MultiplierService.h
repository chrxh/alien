#pragma once

#include <Base/Singleton.h>

#include <EngineInterface/DescEditService.h>
#include <EngineInterface/Descs.h>

// Multiplies the selection including its connected cell networks and selects the result
class MultiplierService
{
    MAKE_SINGLETON(MultiplierService);

public:
    struct Result
    {
        ContentDesc origSelection;
        bool overlappingCheckSuccessful = true;
    };
    Result multiplyInGrid(DescEditService::GridMultiplyParameters const& parameters) const;
    Result multiplyRandomly(DescEditService::RandomMultiplyParameters const& parameters) const;

    void undo(ContentDesc const& origSelection) const;
};
