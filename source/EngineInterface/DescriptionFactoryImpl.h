#pragma once

#include "DescriptionFactory.h"

class DescriptionFactoryImpl : public DescriptionFactory
{
public:
    virtual ~DescriptionFactoryImpl() = default;

    ClusterDescription createHexagon(CreateHexagonParameters const& parameters) const override;

    ClusterDescription createUnconnectedDisc(CreateDiscParameters const& parameters) const override;

    void generateBranchNumbers(
        SimulationParameters const& parameters,
        DataDescription& data,
        std::unordered_set<uint64_t> const& cellIds) const override;

    void randomizeCellFunctions(
        SimulationParameters const& parameters,
        DataDescription& data,
        std::unordered_set<uint64_t> const& cellIds) const override;

    void preserveCellConnections(
        SimulationParameters const& parameters,
        DataDescription& data,
        std::unordered_set<uint64_t> const& cellIds) const override;
};