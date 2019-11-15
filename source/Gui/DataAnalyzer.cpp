#include <boost/range/adaptors.hpp>
#include <QMessageBox>

#include "ModelBasic/SimulationAccess.h"
#include "ModelBasic/Descriptions.h"

#include "Notifier.h"
#include "DataRepository.h"
#include "DataAnalyzer.h"

DataAnalyzer::DataAnalyzer(QObject* parent /*= nullptr*/)
    : QObject(parent)
{}

void DataAnalyzer::init(SimulationAccess* access, DataRepository* repository, Notifier* notifier)
{
    SET_CHILD(_access, access);
    _repository = repository;
    _notifier = notifier;

    for (auto const& connection : _connections) {
        disconnect(connection);
    }
    _connections.clear();

    _connections.push_back(connect(
        _access, &SimulationAccess::dataReadyToRetrieve, this, &DataAnalyzer::dataFromAccessAvailable, Qt::QueuedConnection));
}

void DataAnalyzer::addMostFrequenceClusterRepresentantToSimulation() const
{
    _access->requireData(ResolveDescription());
}

void DataAnalyzer::dataFromAccessAvailable()
{
    DataDescription data = _access->retrieveData();

    auto const partitionDataByDescription = calcPartitionData(data);

    PartitionData mostFrequentClusterData;
    for (auto const& descAndPartitionData : partitionDataByDescription ) {
        auto const& desc = descAndPartitionData.first;
        auto const& partitionData = descAndPartitionData.second;
        if (partitionData.numberOfElements > mostFrequentClusterData.numberOfElements && desc.hasToken) {
            mostFrequentClusterData = partitionData;
        }
    }
    
    if (mostFrequentClusterData.numberOfElements > 0) {
        _repository->addAndSelectData(DataDescription().addCluster(mostFrequentClusterData.representant), { 0, 0 });

        Q_EMIT _notifier->notifyDataRepositoryChanged({
            Receiver::DataEditor, Receiver::Simulation, Receiver::VisualEditor, Receiver::ActionController
        }, UpdateDescription::All);

        QMessageBox msgBox;
        msgBox.setText(QString("%1 exemplars found.").arg(mostFrequentClusterData.numberOfElements));
        msgBox.exec();
    }
    else {
        QMessageBox msgBox;
        msgBox.setText(QString("No exemplars found."));
        msgBox.exec();
    }
}

auto DataAnalyzer::calcPartitionData(DataDescription const& data) const
    -> std::map<ClusterAnalysisDescription, PartitionData>
{
    std::map<ClusterAnalysisDescription, PartitionData> result;

    if (auto const& clusters = data.clusters) {
        for (auto const& cluster : *clusters) {
            auto const clusterAnalysisData = getAnalysisDescription(cluster);
            auto& partitionData = result[clusterAnalysisData];
            if (1 == ++partitionData.numberOfElements) {
                partitionData.representant = cluster;
            }
        }
    }
    return result;
}

auto DataAnalyzer::getAnalysisDescription(ClusterDescription const& cluster) const -> ClusterAnalysisDescription
{
    ClusterAnalysisDescription result;
    result.hasToken = false;
    if (auto const& cells = cluster.cells) {
        for (auto const& cell : *cells) {
            CellAnalysisDescription cellAnalysisData;
            cellAnalysisData.maxConnections = *cell.maxConnections;
            cellAnalysisData.numConnections = cell.connectingCells->size();
            cellAnalysisData.tokenBlocked = *cell.tokenBlocked;
            cellAnalysisData.tokenBranchNumber = *cell.tokenBranchNumber;

            CellFeatureAnalysisDescription featureAnalysisData;
            featureAnalysisData.cellFunction = cell.cellFeature->type;
            cellAnalysisData.feature = featureAnalysisData;

            result.cells.emplace_back(cellAnalysisData);
            if (cell.tokens && cell.tokens->size() > 0) {
                result.hasToken = true;
            }
        }
    }
    return result;
}
