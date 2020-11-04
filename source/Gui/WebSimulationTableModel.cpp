#include "WebSimulationTableModel.h"

#include <QString>
#include <QVariant>

int WebSimulationTableModel::rowCount(const QModelIndex &parent /*= QModelIndex()*/) const
{
    return _simulationInfos.size();
}

int WebSimulationTableModel::columnCount(const QModelIndex &parent /*= QModelIndex()*/) const
{
    return 5;
}

QVariant WebSimulationTableModel::data(const QModelIndex &index, int role) const
{
    if (!index.isValid()) {
        return QVariant();
    }

    if (role == Qt::DisplayRole) {
        if (index.row() >= _simulationInfos.size()) {
            return QVariant();
        }
        auto const simulationInfo = _simulationInfos.at(index.row());
        switch (index.column()) {
        case 0: {
            return QVariant(simulationInfo.isActive);
        } break;
        case 1: {
            return QVariant(QString::fromStdString(simulationInfo.simulationName));
        } break;
        case 2: {
            return QVariant(QString::fromStdString(simulationInfo.userName));
        } break;
        case 3: {
            return QVariant();
        } break;
        case 4: {
            return QVariant(simulationInfo.timestep);
        } break;
        default:
        return QVariant();
        }
    }
    else {
        return QVariant();
    }
}

QVariant WebSimulationTableModel::headerData(int section, Qt::Orientation orientation, int role /*= Qt::DisplayRole*/) const
{
    return QVariant(QString::fromStdString("Header"));
}

void WebSimulationTableModel::setSimulationInfos(vector<SimulationInfo> const & value)
{
    beginResetModel();
    _simulationInfos = value;
    endResetModel();
}
