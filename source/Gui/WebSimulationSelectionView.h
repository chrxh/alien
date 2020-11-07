#pragma once

#include <QDialog>
#include <QItemSelection>

#include "Definitions.h"

namespace Ui {
    class WebSimulationSelectionView;
}

class WebSimulationSelectionView : public QDialog
{
    Q_OBJECT

public:
    WebSimulationSelectionView(
        WebSimulationSelectionController* controller, 
        WebSimulationTableModel* model, 
        QWidget* parent = nullptr);
    virtual ~WebSimulationSelectionView();

    int getIndexOfSelectedSimulation() const;

private:

    Q_SLOT void simulationSelectionChanged(QItemSelection const& selected, QItemSelection const& deselected);

    Ui::WebSimulationSelectionView *_ui;
    WebSimulationSelectionController* _controller;
};

