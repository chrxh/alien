#pragma once

#include <QAbstractTableModel>

#include "Web/SimulationInfo.h"

class WebSimulationTableModel
    : public QAbstractTableModel
{
    Q_OBJECT

public:
    WebSimulationTableModel(QObject *parent = nullptr)
        : QAbstractTableModel(parent) {}

    int rowCount(const QModelIndex &parent = QModelIndex()) const override;
    int columnCount(const QModelIndex &parent = QModelIndex()) const override;
    QVariant data(const QModelIndex &index, int role) const override;
    QVariant headerData(int section, Qt::Orientation orientation,
        int role = Qt::DisplayRole) const override;

    void setSimulationInfos(vector<SimulationInfo> const& value);

private:
    vector<SimulationInfo> _simulationInfos;
};