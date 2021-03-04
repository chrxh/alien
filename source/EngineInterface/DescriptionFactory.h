#pragma once

#include "Descriptions.h"

class ENGINEINTERFACE_EXPORT DescriptionFactory
{
public:
    virtual ~DescriptionFactory() = default;

    struct CreateHexagonParameters
    {
        MEMBER_DECLARATION(CreateHexagonParameters, int, layers, 1);
        MEMBER_DECLARATION(CreateHexagonParameters, double, cellDistance, 1.0);
        MEMBER_DECLARATION(CreateHexagonParameters, double, cellEnergy, 100.0);
        MEMBER_DECLARATION(CreateHexagonParameters, QVector2D, centerPosition, QVector2D());
        MEMBER_DECLARATION(CreateHexagonParameters, double, angle, 0.0);
        MEMBER_DECLARATION(CreateHexagonParameters, int, colorCode, 1);
    };
    virtual ClusterDescription createHexagon(CreateHexagonParameters const& parameters) const = 0;

    struct CreateCircleParameters
    {
        MEMBER_DECLARATION(CreateCircleParameters, int, outerRadius, 1);
        MEMBER_DECLARATION(CreateCircleParameters, int, innerRadius, 1);
        MEMBER_DECLARATION(CreateCircleParameters, double, cellDistance, 1.0);
        MEMBER_DECLARATION(CreateCircleParameters, double, cellEnergy, 100.0);
        MEMBER_DECLARATION(CreateCircleParameters, int, maxConnections, 6);
        MEMBER_DECLARATION(CreateCircleParameters, QVector2D, centerPosition, QVector2D());
        MEMBER_DECLARATION(CreateCircleParameters, int, colorCode, 1);
    };
    virtual ClusterDescription createUnconnectedCircle(CreateCircleParameters const& parameters)
        const = 0;
};