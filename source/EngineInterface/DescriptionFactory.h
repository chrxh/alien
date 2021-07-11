#pragma once

#include "Base/NumberGenerator.h"

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
        MEMBER_DECLARATION(CreateHexagonParameters, QVector2D, velocity, QVector2D());
        MEMBER_DECLARATION(CreateHexagonParameters, int, maxConnections, 6);
        MEMBER_DECLARATION(CreateHexagonParameters, int, colorCode, 1);
    };
    virtual ClusterDescription createHexagon(
        CreateHexagonParameters const& parameters,
        NumberGenerator* numberGenerator) const = 0;

    struct CreateRectParameters
    {
        MEMBER_DECLARATION(CreateRectParameters, IntVector2D, size, IntVector2D({1, 1}));
        MEMBER_DECLARATION(CreateRectParameters, double, cellDistance, 1.0);
        MEMBER_DECLARATION(CreateRectParameters, double, cellEnergy, 100.0);
        MEMBER_DECLARATION(CreateRectParameters, QVector2D, centerPosition, QVector2D());
        MEMBER_DECLARATION(CreateRectParameters, QVector2D, velocity, QVector2D());
        MEMBER_DECLARATION(CreateRectParameters, int, maxConnections, 6);
        MEMBER_DECLARATION(CreateRectParameters, int, colorCode, 1);
    };
    virtual ClusterDescription createRect(CreateRectParameters const& parameters, NumberGenerator* numberGenerator)
        const = 0;

    struct CreateDiscParameters
    {
        MEMBER_DECLARATION(CreateDiscParameters, int, outerRadius, 1);
        MEMBER_DECLARATION(CreateDiscParameters, int, innerRadius, 1);
        MEMBER_DECLARATION(CreateDiscParameters, double, cellDistance, 1.0);
        MEMBER_DECLARATION(CreateDiscParameters, double, cellEnergy, 100.0);
        MEMBER_DECLARATION(CreateDiscParameters, int, maxConnections, 6);
        MEMBER_DECLARATION(CreateDiscParameters, QVector2D, centerPosition, QVector2D());
        MEMBER_DECLARATION(CreateDiscParameters, int, colorCode, 1);
    };
    virtual ClusterDescription createUnconnectedDisc(CreateDiscParameters const& parameters)
        const = 0;

    virtual void generateBranchNumbers(
        SimulationParameters const& parameters,
        DataDescription& data,
        std::unordered_set<uint64_t> const& cellIds) const = 0;

    virtual void randomizeCellFunctions(
        SimulationParameters const& parameters,
        DataDescription& data,
        std::unordered_set<uint64_t> const& cellIds) const = 0;

    virtual void removeFreeCellConnections(
        SimulationParameters const& parameters,
        DataDescription& data,
        std::unordered_set<uint64_t> const& cellIds) const = 0;
};