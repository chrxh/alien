#include <iostream>
#include <fstream>
#include <QFileDialog>
#include <QMessageBox>

#include "Base/ServiceLocator.h"
#include "ModelBasic/SymbolTable.h"
#include "ModelBasic/ModelBasicBuilderFacade.h"
#include "ModelBasic/Serializer.h"
#include "ModelBasic/SerializationHelper.h"
#include "Gui/Settings.h"

#include "SymbolTableDialog.h"
#include "ui_symboltabledialog.h"

SymbolTableDialog::SymbolTableDialog(SymbolTable const* symbolTable, Serializer* serializer, QWidget *parent)
	: QDialog(parent)
	, ui(new Ui::SymbolTableDialog)
	, _symbolTable(symbolTable->clone(parent))
	, _serializer(serializer)
{
    ui->setupUi(this);
    setFont(GuiSettings::getGlobalFont());

    //create headers
    ui->tableWidget->horizontalHeader()->resizeSection(0, 400);
    ui->tableWidget->horizontalHeaderItem(0)->setFont(GuiSettings::getGlobalFont());
    ui->tableWidget->horizontalHeaderItem(0)->setTextAlignment(Qt::AlignLeft);
    ui->tableWidget->horizontalHeader()->resizeSection(1, 100);
    ui->tableWidget->horizontalHeaderItem(1)->setFont(GuiSettings::getGlobalFont());
    ui->tableWidget->horizontalHeaderItem(1)->setTextAlignment(Qt::AlignLeft);

    updateWidgetsFromSymbolTable();

    //connections
    connect(ui->tableWidget, SIGNAL(itemSelectionChanged()), this, SLOT(itemSelectionChanged()));
    connect(ui->addButton, SIGNAL(clicked()), this, SLOT(addButtonClicked()));
    connect(ui->delButton, SIGNAL(clicked()), this, SLOT(delButtonClicked()));
    connect(ui->defaultButton, SIGNAL(clicked()), this, SLOT(defaultButtonClicked()));
    connect(ui->loadButton, SIGNAL(clicked()), this, SLOT(loadButtonClicked()));
    connect(ui->saveButton, SIGNAL(clicked()), this, SLOT(saveButtonClicked()));
    connect(ui->mergeWithButton, SIGNAL(clicked()), this, SLOT(mergeWithButtonClicked()));
}

SymbolTableDialog::~SymbolTableDialog()
{
    delete ui;
}

SymbolTable* SymbolTableDialog::getSymbolTable()
{
	updateSymbolTableFromWidgets();
	return _symbolTable;
}

void SymbolTableDialog::updateSymbolTableFromWidgets ()
{
    _symbolTable->clear();
    for(int i = 0; i < ui->tableWidget->rowCount(); ++i) {
        QTableWidgetItem* item1 = ui->tableWidget->item(i, 0);
        QTableWidgetItem* item2 = ui->tableWidget->item(i, 1);
        _symbolTable->addEntry(item1->text().toStdString(), item2->text().toStdString());
    }
}

void SymbolTableDialog::updateWidgetsFromSymbolTable()
{
    int row = ui->tableWidget->rowCount();
    for(int i = 0; i < row; ++i)
        ui->tableWidget->removeRow(0);

    //create entries in the table
    row = 0;
	for(auto const& keyAndValue : _symbolTable->getEntries()) {

        //create new row in table
        ui->tableWidget->insertRow(row);
        ui->tableWidget->setVerticalHeaderItem(row, new QTableWidgetItem(""));
        QTableWidgetItem* item = new QTableWidgetItem(QString::fromStdString(keyAndValue.first));
        ui->tableWidget->setItem(row, 0, item);
        item = new QTableWidgetItem(QString::fromStdString(keyAndValue.second));
        ui->tableWidget->setItem(row, 1, item);
        row++;
    }
}

void SymbolTableDialog::itemSelectionChanged ()
{
    //items selected?
    if( ui->tableWidget->selectedItems().empty() ) {
        ui->delButton->setEnabled(false);
    }
    else {
        ui->delButton->setEnabled(true);
    }
}

void SymbolTableDialog::addButtonClicked ()
{
    //create new row in table
    int row = ui->tableWidget->rowCount();
    ui->tableWidget->insertRow(row);
    ui->tableWidget->setVerticalHeaderItem(row, new QTableWidgetItem(""));
    QTableWidgetItem* item = new QTableWidgetItem("");
    ui->tableWidget->setItem(row, 0, item);
    ui->tableWidget->setItem(row, 1, new QTableWidgetItem(""));
    ui->tableWidget->editItem(item);
    ui->tableWidget->setCurrentCell(row,0);

    //activate del button
    ui->delButton->setEnabled(true);
}

void SymbolTableDialog::delButtonClicked ()
{
    while( !ui->tableWidget->selectedItems().isEmpty() ) {
        QList< QTableWidgetItem* > items = ui->tableWidget->selectedItems();
        int row = items.at(0)->row();
        ui->tableWidget->removeRow(row);
    }
    if( ui->tableWidget->rowCount() == 0 )
        ui->delButton->setEnabled(false);
}

void SymbolTableDialog::defaultButtonClicked ()
{
	ModelBasicBuilderFacade* facade = ServiceLocator::getInstance().getService<ModelBasicBuilderFacade>();

	_symbolTable = facade->getDefaultSymbolTable();
    updateWidgetsFromSymbolTable();
}

void SymbolTableDialog::loadButtonClicked ()
{
    QString filename = QFileDialog::getOpenFileName(this, "Load Symbol Table", "", "Alien Symbol Table(*.sym)");
	if (!filename.isEmpty()) {
		auto origSymbolTable = _symbolTable;
		if (SerializationHelper::loadFromFile<SymbolTable*>(filename.toStdString(), [&](string const& data) { return _serializer->deserializeSymbolTable(data); }, _symbolTable)) {
			delete origSymbolTable;
			updateWidgetsFromSymbolTable();
		}
		else {
			QMessageBox msgBox(QMessageBox::Critical, "Error", "An error occurred. The specified symbol table could not loaded.");
			msgBox.exec();
		}
    }
}

void SymbolTableDialog::saveButtonClicked ()
{
    QString filename = QFileDialog::getSaveFileName(this, "Save Symbol Table", "", "Alien Symbol Table (*.sym)");
    if( !filename.isEmpty() ) {
		updateSymbolTableFromWidgets();
		if (!SerializationHelper::saveToFile(filename.toStdString(), [&]() { return _serializer->serializeSymbolTable(_symbolTable); })) {
			QMessageBox msgBox(QMessageBox::Critical, "Error", "An error occurred. The symbol table could not saved.");
			msgBox.exec();
			return;
		}
    }
}

void SymbolTableDialog::mergeWithButtonClicked ()
{
    QString filename = QFileDialog::getOpenFileName(this, "Load Symbol Table", "", "Alien Symbol Table(*.sym)");
    if( !filename.isEmpty() ) {

		SymbolTable* symbolTableToMerge;
		if (SerializationHelper::loadFromFile<SymbolTable*>(filename.toStdString(), [&](string const& data) { return _serializer->deserializeSymbolTable(data); }, symbolTableToMerge)) {
			_symbolTable->mergeEntries(*symbolTableToMerge);
			delete symbolTableToMerge;
			updateWidgetsFromSymbolTable();
		}
		else {
			QMessageBox msgBox(QMessageBox::Critical, "Error", "An error occurred. The specified symbol table could not loaded.");
			msgBox.exec();
		}
    }
}
