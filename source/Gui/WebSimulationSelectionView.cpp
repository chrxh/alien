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
    , ui(new Ui::WebSimulationSelectionView)
    , _controller(controller)
{
    ui->setupUi(this);
    setFont(GuiSettings::getGlobalFont());

    ui->webSimulationTableView->setModel(model);
    ui->webSimulationTableView->setAlternatingRowColors(true);
    ui->webSimulationTableView->horizontalHeader()->setStretchLastSection(true);
    ui->webSimulationTableView->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);

    connect(ui->refreshButton, &QPushButton::clicked, _controller, &WebSimulationSelectionController::refresh);
}


WebSimulationSelectionView::~WebSimulationSelectionView()
{
    delete ui;
}
