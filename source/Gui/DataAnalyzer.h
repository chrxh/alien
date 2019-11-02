#pragma once

#include <QObject>

#include "ModelBasic/Descriptions.h"

#include "Definitions.h"

class DataAnalyzer : public QObject
{
    Q_OBJECT
public:
    DataAnalyzer(QObject* parent = nullptr);
    virtual ~DataAnalyzer() = default;

    void init(SimulationAccess* access, DataRepository* repository, Notifier* notifier);

    void addMostFrequenceClusterRepresentantToSimulation() const;

private:
    Q_SLOT void dataFromAccessAvailable();

    struct CellFeatureAnalysisDescription
    {
        int cellFunction;

        bool operator<(CellFeatureAnalysisDescription const& other) const { return cellFunction < other.cellFunction; }
        bool operator!=(CellFeatureAnalysisDescription const& other) const
        {
            return cellFunction != other.cellFunction;
        }
    };
    struct CellAnalysisDescription
    {
        int maxConnections;
        int numConnections;
        bool tokenBlocked;
        int tokenBranchNumber;
        CellFeatureAnalysisDescription feature;
        bool operator<(CellAnalysisDescription const& other) const
        {
            if (maxConnections != other.maxConnections) {
                return maxConnections < other.maxConnections;
            }
            if (numConnections != other.numConnections) {
                return numConnections < other.numConnections;
            }
            if (tokenBlocked != other.tokenBlocked) {
                return tokenBlocked < other.tokenBlocked;
            }
            if (tokenBranchNumber != other.tokenBranchNumber) {
                return tokenBranchNumber < other.tokenBranchNumber;
            }
            if (feature != other.feature) {
                return feature < other.feature;
            }
            return false;
        }
    };
    struct ClusterAnalysisDescription
    {
        bool hasToken;
        std::vector<CellAnalysisDescription> cells;

        bool operator<(ClusterAnalysisDescription const& other) const { return cells < other.cells; }
    };
    struct PartitionData
    {
        int numberOfElements = 0;
        ClusterDescription representant;
    };

    std::map<ClusterAnalysisDescription, PartitionData> calcPartitionData(DataDescription const& data) const;

    ClusterAnalysisDescription getAnalysisDescription(ClusterDescription const& cluster) const;

private:
    list<QMetaObject::Connection> _connections;

    SimulationAccess* _access = nullptr;
    DataRepository* _repository = nullptr;
    Notifier* _notifier = nullptr;
};
