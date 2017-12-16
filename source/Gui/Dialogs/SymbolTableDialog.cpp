#include <iostream>
#include <fstream>
#include <QFileDialog>
#include <QMessageBox>

#include "Base/ServiceLocator.h"
#include "Model/Api/Settings.h"
#include "Model/Api/SymbolTable.h"
#include "Model/Api/ModelBuilderFacade.h"
#include "Model/Api/Serializer.h"
#include "Gui/Settings.h"

#include "SymbolTableDialog.h"
#include "ui_symboltabledialog.h"

SymbolTableDialog::SymbolTableDialog(SymbolTable* symbolTable, Serializer* serializer, QWidget *parent)
	: QDialog(parent)
	, ui(new Ui::SymbolTableDialog)
	, _symbolTable(symbolTable->clone())
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

SymbolTable * SymbolTableDialog::loadSymbolTable(string filename)
{
	std::ifstream stream(filename, std::ios_base::in | std::ios_base::binary);

	size_t size;
	string data;

	stream.read(reinterpret_cast<char*>(&size), sizeof(size_t));
	data.resize(size);
	stream.read(&data[0], size);
	stream.close();

	if (stream.fail()) {
		return nullptr;
	}

	try {
		return _serializer->deserializeSymbolTable(data);
	}
	catch (...) {
		return nullptr;
	}
}

bool SymbolTableDialog::saveSymbolTable(string filename, SymbolTable * symbolTable)
{
	try {
		std::ofstream stream(filename, std::ios_base::out | std::ios_base::binary);
		string const& data = _serializer->serializeSymbolTable(_symbolTable);;
		size_t dataSize = data.size();
		stream.write(reinterpret_cast<char*>(&dataSize), sizeof(size_t));
		stream.write(&data[0], data.size());
		stream.close();
		if (stream.fail()) {
			return false;
		}
	}
	catch (...) {
		return false;
	}
	return true;
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
	ModelBuilderFacade* facade = ServiceLocator::getInstance().getService<ModelBuilderFacade>();

	_symbolTable = facade->buildDefaultSymbolTable();
    updateWidgetsFromSymbolTable();
}

void SymbolTableDialog::loadButtonClicked ()
{
    QString filename = QFileDialog::getOpenFileName(this, "Load Symbol Table", "", "Alien Symbol Table(*.sym)");
	if (!filename.isEmpty()) {
		SymbolTable* symbolTable = loadSymbolTable(filename.toStdString());
		if (symbolTable) {
			delete _symbolTable;
			_symbolTable = symbolTable;
			updateWidgetsFromSymbolTable();
		}
        else {
            QMessageBox msgBox(QMessageBox::Critical,"Error", "An error occurred. The specified symbol table could not loaded.");
            msgBox.exec();
        }
    }
}

void SymbolTableDialog::saveButtonClicked ()
{
    QString filename = QFileDialog::getSaveFileName(this, "Save Symbol Table", "", "Alien Symbol Table (*.sym)");
    if( !filename.isEmpty() ) {
		updateSymbolTableFromWidgets();
		if (!saveSymbolTable(filename.toStdString(), _symbolTable)) {
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

		SymbolTable* symbolTable = loadSymbolTable(filename.toStdString());
		if (symbolTable) {
			_symbolTable->mergeEntries(*symbolTable);
			delete symbolTable;
			updateWidgetsFromSymbolTable();
		}
		else {
			QMessageBox msgBox(QMessageBox::Critical, "Error", "An error occurred. The specified symbol table could not loaded.");
			msgBox.exec();
		}
    }
}
