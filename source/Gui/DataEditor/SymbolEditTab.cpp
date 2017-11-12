#include <QScrollBar>

#include "Model/Api/SymbolTable.h"
#include "Gui/Settings.h"

#include "DataEditorModel.h"
#include "DataEditorController.h"
#include "SymbolEditTab.h"
#include "ui_SymbolEditTab.h"

SymbolEditTab::SymbolEditTab(QWidget *parent)
	: QWidget(parent), ui(new Ui::SymbolEditTab)
{
    ui->setupUi(this);

    setStyleSheet("background-color: #000000");
    ui->tableWidget->setStyleSheet(TABLE_STYLESHEET);
    ui->tableWidget->verticalScrollBar()->setStyleSheet(SCROLLBAR_STYLESHEET);
    ui->addSymbolButton->setStyleSheet(BUTTON_STYLESHEET);
    ui->delSymbolButton->setStyleSheet(BUTTON_STYLESHEET);

    QPalette p = ui->addSymbolButton->palette();
    p.setColor(QPalette::ButtonText, BUTTON_TEXT_COLOR);
    ui->addSymbolButton->setPalette(p);
    ui->delSymbolButton->setPalette(p);

    ui->tableWidget->setFont(GuiSettings::getGlobalFont());

    ui->tableWidget->horizontalHeader()->resizeSection(0, 285);
    ui->tableWidget->horizontalHeader()->resizeSection(1, 55);

    connect(ui->addSymbolButton, &QPushButton::clicked, this, &SymbolEditTab::addSymbolButtonClicked);
    connect(ui->delSymbolButton, &QPushButton::clicked, this, &SymbolEditTab::delSymbolButtonClicked);
    connect(ui->tableWidget, &QTableWidget::itemSelectionChanged, this, &SymbolEditTab::itemSelectionChanged);
}

SymbolEditTab::~SymbolEditTab()
{
    delete ui;
}

void SymbolEditTab::init(DataEditorModel * model, DataEditorController * controller)
{
	_model = model;
	_controller = controller;
}

void SymbolEditTab::updateDisplay()
{
	disconnect(ui->tableWidget, &QTableWidget::itemChanged, this, &SymbolEditTab::itemContentChanged);

    //delete rows in the table
    while( ui->tableWidget->rowCount() > 0 )
        ui->tableWidget->removeRow(0);

    //create entries in the table
    int i = 0;
	for(pair<string, string>const& keyAndValue : _model->getSymbolTable()->getEntries()) {

        //create new row in table
        ui->tableWidget->insertRow(i);

        //set values
        string key = keyAndValue.first;
		string value = keyAndValue.second;
        ui->tableWidget->setItem(i, 0, new QTableWidgetItem(QString::fromStdString(key)));
        ui->tableWidget->setItem(i, 1, new QTableWidgetItem(QString::fromStdString(value)));
        ui->tableWidget->item(i, 0)->setTextColor(CELL_EDIT_DATA_COLOR1);
        ui->tableWidget->item(i, 1)->setTextColor(CELL_EDIT_DATA_COLOR1);
        ++i;
    }

    //set del button
    if( ui->tableWidget->rowCount() > 0 )
        ui->delSymbolButton->setEnabled(true);
    else
        ui->delSymbolButton->setEnabled(false);

    connect(ui->tableWidget, &QTableWidget::itemChanged, this, &SymbolEditTab::itemContentChanged);
}

void SymbolEditTab::addSymbolButtonClicked()
{
	disconnect(ui->tableWidget, &QTableWidget::itemChanged, this, &SymbolEditTab::itemContentChanged);

	//create new row in table
	int row = ui->tableWidget->rowCount();
	ui->tableWidget->insertRow(row);
	ui->tableWidget->setVerticalHeaderItem(row, new QTableWidgetItem(""));
	{
		QTableWidgetItem* item = new QTableWidgetItem("");
		item->setTextColor(CELL_EDIT_DATA_COLOR1);
		ui->tableWidget->setItem(row, 0, item);
		ui->tableWidget->editItem(item);
	}
	{
		QTableWidgetItem* item = new QTableWidgetItem("");
		item->setTextColor(CELL_EDIT_DATA_COLOR1);
		ui->tableWidget->setItem(row, 1, item);
		ui->tableWidget->setCurrentCell(row, 0);
	}

	//activate del button
	ui->delSymbolButton->setEnabled(true);

	connect(ui->tableWidget, &QTableWidget::itemChanged, this, &SymbolEditTab::itemContentChanged);
}

void SymbolEditTab::delSymbolButtonClicked ()
{
	auto symbols = _model->getSymbolTable();
	while (!ui->tableWidget->selectedItems().isEmpty()) {
        QList< QTableWidgetItem* > items = ui->tableWidget->selectedItems();
        int row = items.at(0)->row();
        QString key = ui->tableWidget->item(row, 0)->text();
		symbols->delEntry(key.toStdString());
        ui->tableWidget->removeRow(row);
    }
    if( ui->tableWidget->rowCount() == 0 )
        ui->delSymbolButton->setEnabled(false);

	_controller->notificationFromSymbolTab();
}

void SymbolEditTab::itemSelectionChanged ()
{
    //items selected?
    if( ui->tableWidget->selectedItems().empty() ) {
        ui->delSymbolButton->setEnabled(false);
    }
    else {
        ui->delSymbolButton->setEnabled(true);
    }
}

void SymbolEditTab::itemContentChanged (QTableWidgetItem* item)
{
    //clear and update complete symbol table
	auto symbols = _model->getSymbolTable();
	symbols->clear();
    for(int i = 0; i < ui->tableWidget->rowCount(); ++i) {

        //obtain key and value text
        QString key = ui->tableWidget->item(i, 0)->text();
        QString value = ui->tableWidget->item(i, 1)->text();
		symbols->addEntry(key.toStdString(), value.toStdString());
    }

	_controller->notificationFromSymbolTab();
}



