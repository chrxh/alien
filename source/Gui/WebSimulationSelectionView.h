#pragma once

#include <QDialog>

#include "Definitions.h"

namespace Ui {
    class WebSimulationSelectionView;
}

class WebSimulationSelectionView : public QDialog
{
    Q_OBJECT

public:
    WebSimulationSelectionView(WebSimulationTableModel* model, QWidget* parent = nullptr);
    virtual ~WebSimulationSelectionView();

private:
    Ui::WebSimulationSelectionView *ui;
};

