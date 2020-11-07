
#include "WebSimulationSelectionView.h"

#include "Gui/Settings.h"

#include "WebSimulationSelectionController.h"
#include "WebSimulationTableModel.h"
#include "ui_WebSimulationSelectionView.h"

WebSimulationSelectionView::WebSimulationSelectionView(
    WebSimulationSelectionController* controller, 
    WebSimulationTableModel* model, 
    QWidget *parent)
    : QDialog(parent)
    , _ui(new Ui::WebSimulationSelectionView)
    , _controller(controller)
{
    _ui->setupUi(this);
    setFont(GuiSettings::getGlobalFont());

    _ui->webSimulationTreeView->setModel(model);
    _ui->webSimulationTreeView->setAlternatingRowColors(true);
    _ui->webSimulationTreeView->setRootIsDecorated(false);
    _ui->webSimulationTreeView->header()->setStretchLastSection(false);
    _ui->webSimulationTreeView->header()->setSectionResizeMode(QHeaderView::ResizeToContents); 
    auto const selectionModel = _ui->webSimulationTreeView->selectionModel();

    connect(_ui->refreshButton, &QPushButton::clicked, _controller, &WebSimulationSelectionController::refresh);
    connect(selectionModel, &QItemSelectionModel::selectionChanged, this, &WebSimulationSelectionView::simulationSelectionChanged);
}


WebSimulationSelectionView::~WebSimulationSelectionView()
{
    delete _ui;
}

int WebSimulationSelectionView::getIndexOfSelectedSimulation() const
{
    auto const selectionModel = _ui->webSimulationTreeView->selectionModel();
    return selectionModel->selectedRows().front().row();
}

void WebSimulationSelectionView::simulationSelectionChanged(QItemSelection const& selected, QItemSelection const& deselected)
{
    auto const anyRowSelected = selected.indexes().size() > 0;
    _ui->okButton->setEnabled(anyRowSelected);
}
