#pragma once

#include "EngineInterface/Descriptions.h"
#include "Definitions.h"

class _PatternAnalysisDialog
{
public:
    _PatternAnalysisDialog(SimulationController const& simController);
    ~_PatternAnalysisDialog();

    void process();
    void show();

private:
    void saveRepetitiveActiveClustersToFiles(std::string const& filename);

    struct CellAnalysisDescription
    {
        int maxConnections;
        int numConnections;
        bool constructionState;
        std::optional<int> inputExecutionOrderNumber;
        bool outputBlocked;
        int executionOrderNumber;
        int color;
        int cellFunction;

        bool operator==(CellAnalysisDescription const& other) const
        {
            return maxConnections == other.maxConnections && numConnections == other.numConnections && constructionState == other.constructionState
                && inputExecutionOrderNumber == other.inputExecutionOrderNumber && outputBlocked == other.outputBlocked && executionOrderNumber == other.executionOrderNumber
                && cellFunction == other.cellFunction
                && color == other.color;
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
            if (inputExecutionOrderNumber != other.inputExecutionOrderNumber) {
                return inputExecutionOrderNumber < other.inputExecutionOrderNumber;
            }
            if (outputBlocked != other.outputBlocked) {
                return outputBlocked < other.outputBlocked;
            }
            if (executionOrderNumber != other.executionOrderNumber) {
                return executionOrderNumber < other.executionOrderNumber;
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
    SimulationController _simController;

    std::string _startingPath;
};
