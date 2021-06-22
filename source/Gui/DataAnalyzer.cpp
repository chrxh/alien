#include <boost/range/adaptors.hpp>
#include <boost/range/adaptor/indexed.hpp>

#include <QMessageBox>

#include "EngineInterface/SimulationAccess.h"
#include "EngineInterface/SerializationHelper.h"
#include "EngineInterface/Serializer.h"
#include "EngineInterface/Descriptions.h"

#include "Notifier.h"
#include "DataRepository.h"
#include "DataAnalyzer.h"

DataAnalyzer::DataAnalyzer(QObject* parent /*= nullptr*/)
    : QObject(parent)
{}

void DataAnalyzer::init(
    SimulationAccess* access,
    DataRepository* repository,
    Notifier* notifier,
    Serializer* serializer)
{
    SET_CHILD(_access, access);
    _repository = repository;
    _notifier = notifier;
    _serializer = serializer;

    for (auto const& connection : _connections) {
        disconnect(connection);
    }
    _connections.clear();

    _connections.push_back(connect(
        _access, &SimulationAccess::dataReadyToRetrieve, this, &DataAnalyzer::dataFromAccessAvailable, Qt::QueuedConnection));
}

void DataAnalyzer::saveRepetitiveActiveClustersToFiles(std::string const& folder)
{
    _folder = folder;
    _access->requireData(ResolveDescription());
}

void DataAnalyzer::dataFromAccessAvailable()
{
    DataDescription data = _access->retrieveData();

    auto const partitionClassDataByDescription = calcPartitionData(data);

    std::ofstream file;
    file.open(_folder + "/result.txt", std::ios_base::out);

    int sum = 0;
    std::vector<PartitionClassData> partitionData;
    for (auto const& [analysisDesc, partitionClassData] : partitionClassDataByDescription) {
        if (partitionClassData.numberOfElements > 1 && analysisDesc.hasToken) {
            partitionData.emplace_back(partitionClassData);
        }
    }
    std::sort(partitionData.begin(), partitionData.end());

    file << "number of repetitive active clusters: " << partitionData.size() << std::endl << std::endl;
    for (auto const& [index, partitionClassData] :
         partitionData | boost::adaptors::reversed | boost::adaptors::indexed(1)) {

        file << "cluster " << index << ": " << partitionClassData.numberOfElements << " exemplars" << std::endl;
        std::string filename = _folder + "/cluster" + QString("%1").arg(index, 5, 10, QChar('0')).toStdString() + ".col";

        DataDescription pattern;
        pattern.clusters = std::vector<ClusterDescription>{partitionClassData.representant};
        SerializationHelper::saveToFile(filename, [&]() { return _serializer->serializeDataDescription(pattern); });
    }

    QMessageBox msgBox;
    msgBox.setWindowTitle("Analysis result");
    msgBox.setText(QString("%1 repetitive active clusters found. Summary saved to %2/result.txt.")
                       .arg(partitionData.size())
                       .arg(QString::fromStdString(_folder)));
    msgBox.exec();
}

auto DataAnalyzer::calcPartitionData(DataDescription const& data) const
    -> std::map<ClusterAnalysisDescription, PartitionClassData>
{
    std::map<ClusterAnalysisDescription, PartitionClassData> result;

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

    std::map<uint64_t, CellAnalysisDescription> cellAnalysisDescById;
    auto insertCellAnalysisDescription = [&cellAnalysisDescById](CellDescription const& cell) {
        if (cellAnalysisDescById.find(cell.id) == cellAnalysisDescById.end()) {
            CellAnalysisDescription result;
            result.maxConnections = *cell.maxConnections;
            result.numConnections = cell.connectingCells->size();
            result.tokenBlocked = *cell.tokenBlocked;
            result.tokenBranchNumber = *cell.tokenBranchNumber;
//            result.color = cell.metadata->color;

            CellFeatureAnalysisDescription featureAnalysisData;
            featureAnalysisData.cellFunction = cell.cellFeature->getType();
            result.feature = featureAnalysisData;

            cellAnalysisDescById.insert_or_assign(cell.id, result);
        }
    };

    if (auto const& cells = cluster.cells) {

        std::map<uint64_t, int> cellDescIndexById;
        for (auto const& [index, cell] : *cells | boost::adaptors::indexed(0)) {
            cellDescIndexById.insert_or_assign(cell.id, index);
        }

        for (auto const& cell : *cells) {
            insertCellAnalysisDescription(cell);
            for (auto const& connectingCellId : *cell.connectingCells) {
                insertCellAnalysisDescription(cluster.cells->at(cellDescIndexById.at(connectingCellId)));
                result.connectedCells.insert(std::set<CellAnalysisDescription>{
                    cellAnalysisDescById.at(cell.id), cellAnalysisDescById.at(connectingCellId)});
            }
            if (cell.tokens && cell.tokens->size() > 0) {
                result.hasToken = true;
            }
        }
    }
    return result;
}
