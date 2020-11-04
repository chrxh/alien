#include "WebSimulationSelectionView.h"

#include "Gui/Settings.h"

#include "WebSimulationTableModel.h"
#include "ui_WebSimulationSelectionView.h"

WebSimulationSelectionView::WebSimulationSelectionView(WebSimulationTableModel* model, QWidget *parent)
    : QDialog(parent)
    , ui(new Ui::WebSimulationSelectionView)
{
    ui->setupUi(this);
    setFont(GuiSettings::getGlobalFont());

    ui->webSimulationTableView->setModel(model);
}


WebSimulationSelectionView::~WebSimulationSelectionView()
{
    delete ui;
}
