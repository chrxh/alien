#pragma once

#include "DescriptionFactory.h"

class DescriptionFactoryImpl : public DescriptionFactory
{
public:
    virtual ~DescriptionFactoryImpl() = default;

    virtual ClusterDescription createHexagon(CreateHexagonParameters const& parameters) override;
};