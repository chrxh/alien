#include "PatternAnalysisDialog.h"

#include <fstream>

#include <boost/range/adaptor/indexed.hpp>
#include <boost/range/adaptors.hpp>

#include <ImFileDialog.h>

#include "EngineInterface/Descriptions.h"
#include "EngineInterface/Serializer.h"
#include "EngineInterface/SimulationController.h"
#include "GlobalSettings.h"


_PatternAnalysisDialog::_PatternAnalysisDialog(SimulationController const& simController)
    : _simController(simController)
{
    auto path = std::filesystem::current_path();
    if (path.has_parent_path()) {
        path = path.parent_path();
    }
    _startingPath = GlobalSettings::getInstance().getStringState("dialogs.pattern analysis.starting path", path.string());
}

_PatternAnalysisDialog::~_PatternAnalysisDialog()
{
    GlobalSettings::getInstance().setStringState("dialogs.pattern analysis.starting path", _startingPath);
}

void _PatternAnalysisDialog::process()
{
    if (!ifd::FileDialog::Instance().IsDone("PatternAnalysisDialog")) {
        return;
    }
    if (ifd::FileDialog::Instance().HasResult()) {
        auto firstFilename = ifd::FileDialog::Instance().GetResult();
        auto firstFilenameCopy = firstFilename;
        _startingPath = firstFilenameCopy.remove_filename().string();

        saveRepetitiveActiveClustersToFiles(firstFilename.string());
    }
    ifd::FileDialog::Instance().Close();
}

void _PatternAnalysisDialog::show()
{
    ifd::FileDialog::Instance().Save("PatternAnalysisDialog", "Save pattern analysis result", "Analysis result (*.txt){.txt},.*", _startingPath);
}

void _PatternAnalysisDialog::saveRepetitiveActiveClustersToFiles(std::string const& filename)
{
    auto const partitionClassDataByDescription = calcPartitionData();

    std::ofstream file;
    file.open(filename, std::ios_base::out);

    int sum = 0;
    std::vector<PartitionClassData> partitionData;
    for (auto const& [analysisDesc, partitionClassData] : partitionClassDataByDescription) {
        if (partitionClassData.numberOfElements > 1 && analysisDesc.hasToken) {
            partitionData.emplace_back(partitionClassData);
        }
    }
    std::sort(partitionData.begin(), partitionData.end());

    file << "number of repetitive active clusters: " << partitionData.size() << std::endl << std::endl;
    for (auto const& [index, partitionClassData] : partitionData | boost::adaptors::reversed | boost::adaptors::indexed(1)) {

        file << "cluster " << index << ": " << partitionClassData.numberOfElements << " exemplars" << std::endl;

        std::stringstream clusterNameStream;
        clusterNameStream << "cluster" << std::setfill('0') << std::setw(7) << index << ".sim";

        std::filesystem::path clusterFilename(filename);
        clusterFilename.remove_filename();
        clusterFilename /= clusterNameStream.str();

        ClusteredDataDescription pattern;
        pattern.clusters = std::vector<ClusterDescription>{partitionClassData.representant};

        Serializer::serializeContentToFile(clusterFilename.string(), pattern);
    }

/*
    QMessageBox msgBox;
    msgBox.setWindowTitle("Analysis result");
    msgBox.setText(
        QString("%1 repetitive active clusters found. Summary saved to %2/result.txt.").arg(partitionData.size()).arg(QString::fromStdString(_folder)));
    msgBox.exec();
*/
}

auto _PatternAnalysisDialog::calcPartitionData() const -> std::map<ClusterAnalysisDescription, PartitionClassData>
{
    auto data = _simController->getClusteredSimulationData();

    std::map<ClusterAnalysisDescription, PartitionClassData> result;

    for (auto const& cluster : data.clusters) {
        auto const clusterAnalysisData = getAnalysisDescription(cluster);
        auto& partitionData = result[clusterAnalysisData];
        if (1 == ++partitionData.numberOfElements) {
            partitionData.representant = cluster;
        }
    }
    return result;
}

auto _PatternAnalysisDialog::getAnalysisDescription(ClusterDescription const& cluster) const -> ClusterAnalysisDescription
{
    ClusterAnalysisDescription result;
    result.hasToken = false;

    std::map<uint64_t, CellAnalysisDescription> cellAnalysisDescById;
    auto insertCellAnalysisDescription = [&cellAnalysisDescById](CellDescription const& cell) {
        if (cellAnalysisDescById.find(cell.id) == cellAnalysisDescById.end()) {
            CellAnalysisDescription result;
            result.maxConnections = cell.maxConnections;
            result.numConnections = cell.connections.size();
            result.tokenBlocked = cell.tokenBlocked;
            result.tokenBranchNumber = cell.tokenBranchNumber;
            //            result.color = cell.metadata.color;

            CellFeatureAnalysisDescription featureAnalysisData;
            featureAnalysisData.cellFunction = cell.cellFeature.getType();
            result.feature = featureAnalysisData;

            cellAnalysisDescById.insert_or_assign(cell.id, result);
        }
    };

    std::map<uint64_t, int> cellDescIndexById;
    for (auto const& [index, cell] : cluster.cells | boost::adaptors::indexed(0)) {
        cellDescIndexById.insert_or_assign(cell.id, toInt(index));
    }

    for (auto const& cell : cluster.cells) {
        insertCellAnalysisDescription(cell);
        for (auto const& connection : cell.connections) {
            auto connectingCellId = connection.cellId;
            insertCellAnalysisDescription(cluster.cells.at(cellDescIndexById.at(connectingCellId)));
            result.connectedCells.insert(std::set<CellAnalysisDescription>{cellAnalysisDescById.at(cell.id), cellAnalysisDescById.at(connectingCellId)});
        }
        if (cell.tokens.size() > 0) {
            result.hasToken = true;
        }
    }
    return result;
}
