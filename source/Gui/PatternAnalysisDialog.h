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
        int maxConnections;
        int numConnections;
        bool constructionState;
        int color;
        int cellFunction;

        bool operator==(CellAnalysisDescription const& other) const
        {
            return maxConnections == other.maxConnections && numConnections == other.numConnections && constructionState == other.constructionState
                && cellFunction == other.cellFunction && color == other.color;
        }

        bool operator!=(CellAnalysisDescription const& other) const { return !operator==(other); }

        bool operator<(CellAnalysisDescription const& other) const
        {
            if (maxConnections != other.maxConnections) {
                return maxConnections < other.maxConnections;
            }
            if (numConnections != other.numConnections) {
                return numConnections < other.numConnections;
            }
            if (constructionState != other.constructionState) {
                return constructionState < other.constructionState;
            }
            if (cellFunction != other.cellFunction) {
                return cellFunction < other.cellFunction;
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
