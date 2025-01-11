#pragma once

#include "Base/Singleton.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationFacade.h"

#include "Definitions.h"
#include "MainLoopEntity.h"

class PatternAnalysisDialog : public MainLoopEntity<SimulationFacade>
{
    MAKE_SINGLETON(PatternAnalysisDialog);

public:
    void show();

private:
    void init(SimulationFacade simulationFacade) override;
    void process() override;
    void shutdown() override;
    void saveRepetitiveActiveClustersToFiles(std::string const& filename);

    struct CellAnalysisDescription
    {
        int numConnections;
        bool constructionState;
        int color;
        int cellType;

        bool operator==(CellAnalysisDescription const& other) const
        {
            return numConnections == other.numConnections && constructionState == other.constructionState
                && cellType == other.cellType && color == other.color;
        }

        bool operator!=(CellAnalysisDescription const& other) const { return !operator==(other); }

        bool operator<(CellAnalysisDescription const& other) const
        {
            if (numConnections != other.numConnections) {
                return numConnections < other.numConnections;
            }
            if (constructionState != other.constructionState) {
                return constructionState < other.constructionState;
            }
            if (cellType != other.cellType) {
                return cellType < other.cellType;
            }
            if (color != other.color) {
                return color < other.color;
            }
            return false;
        }
    };
    struct ClusterAnalysisDescription
    {
        std::set<std::set<CellAnalysisDescription>> connectedCells;

        bool operator<(ClusterAnalysisDescription const& other) const
        {
            if (connectedCells != other.connectedCells) {
                return connectedCells < other.connectedCells;
            }
            return false;
        }
    };
    struct PartitionClassData
    {
        int numberOfElements = 0;
        ClusterDescription representant;

        bool operator<(PartitionClassData const& other) const { return numberOfElements < other.numberOfElements; };
    };

    std::map<ClusterAnalysisDescription, PartitionClassData> calcPartitionData() const;

    ClusterAnalysisDescription getAnalysisDescription(ClusterDescription const& cluster) const;

private:
    SimulationFacade _simulationFacade;

    std::string _startingPath;
};
