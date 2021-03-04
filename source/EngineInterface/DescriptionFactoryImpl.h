#pragma once

#include "DescriptionFactory.h"

class DescriptionFactoryImpl : public DescriptionFactory
{
public:
    virtual ~DescriptionFactoryImpl() = default;

    ClusterDescription createHexagon(CreateHexagonParameters const& parameters) const override;

    ClusterDescription createUnconnectedCircle(CreateCircleParameters const& parameters)
        const override;
};