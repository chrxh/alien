#pragma once

#include "DescriptionFactory.h"

class DescriptionFactoryImpl : public DescriptionFactory
{
public:
    virtual ~DescriptionFactoryImpl() = default;

    ClusterDescription createHexagon(CreateHexagonParameters const& parameters) const override;

    ClusterDescription createUnconnectedDisc(CreateDiscParameters const& parameters) const override;

    void generateBranchNumbers(DataDescription& data, std::unordered_set<uint64_t> cellIds) const override;
};